#!/usr/bin/env python3
"""Read-only ext4 file extractor: pull chosen files off a Linux SD card from Windows, with no driver, no mount, no deps.

Why: WSL cannot mount a card behind a USB reader (DH-11) and the laptop's C: is too small to image a 29 GB partition. This
reads the raw disk (Windows: \\\\.\\PhysicalDriveN, needs an ADMIN prompt; or any image file) and copies out only the paths
asked for. It never writes to the source: the source is opened 'rb' and only read.

    py -3.9 tools/ext4_extract.py \\\\.\\PhysicalDrive1 --offset-mb 16 --out data/radxa_sd  /etc/NetworkManager  /var/log

Secrets: in extracted TEXT files, values of psk= / password= / wpa-psk / "password:" lines are replaced by <redacted>, so a
Wi-Fi key never lands on the laptop's disk in the clear. Supports extents (any depth), legacy block maps, linear and htree
directories (scanned linearly), fast and slow symlinks. Not supported: inline_data files (reported and skipped).
Python 3.9, standard library only.
"""
from __future__ import annotations

import argparse
import os
import re
import stat
import struct
import sys

SECRET_RE = re.compile(r"^(\s*(?:psk|password|wpa-psk|wep-key\d|private-key-password|leap-password)\s*[=:]\s*).*$",
                       re.IGNORECASE)


class Source:
    """Aligned, cached reads (Windows raw disks only accept sector-aligned reads)."""

    ALIGN = 4096

    def __init__(self, path, base=0):
        self.f = open(path, "rb", buffering=0)
        self.base = int(base)
        self.cache = {}

    def read(self, off, n):
        off += self.base
        out = bytearray()
        a = off - (off % self.ALIGN)
        end = off + n
        while a < end:
            blk = self.cache.get(a)
            if blk is None:
                self.f.seek(a)
                blk = self.f.read(self.ALIGN)
                if len(self.cache) > 8192:
                    self.cache.clear()
                self.cache[a] = blk
            lo = max(off, a) - a
            hi = min(end, a + self.ALIGN) - a
            out += blk[lo:hi]
            if len(blk) < self.ALIGN:
                break
            a += self.ALIGN
        return bytes(out)


class Ext4:
    def __init__(self, src: Source):
        self.src = src
        sb = src.read(1024, 1024)
        (self.inodes_count, blocks_lo, _r, _f, _free_i, self.first_data_block, log_bs) = struct.unpack_from("<7I", sb, 0)
        if struct.unpack_from("<H", sb, 56)[0] != 0xEF53:
            raise ValueError("no ext2/3/4 superblock at this offset (wrong --offset-mb?)")
        self.bs = 1024 << log_bs
        self.blocks_per_group = struct.unpack_from("<I", sb, 32)[0]
        self.ipg = struct.unpack_from("<I", sb, 40)[0]
        rev = struct.unpack_from("<I", sb, 76)[0]
        self.inode_size = struct.unpack_from("<H", sb, 88)[0] if rev >= 1 else 128
        self.incompat = struct.unpack_from("<I", sb, 96)[0]
        is64 = bool(self.incompat & 0x80)
        self.desc_size = (struct.unpack_from("<H", sb, 254)[0] or 64) if is64 else 32
        self.volume_name = sb[120:136].split(b"\0")[0].decode("utf-8", "replace")
        u = sb[0x68:0x78].hex()
        self.uuid = f"{u[:8]}-{u[8:12]}-{u[12:16]}-{u[16:20]}-{u[20:]}"
        self.ro_compat = struct.unpack_from("<I", sb, 100)[0]
        self._gd = {}

    def block(self, n, count=1):
        return self.src.read(n * self.bs, count * self.bs)

    def _inode_table(self, group):
        if group not in self._gd:
            gdt = (self.first_data_block + 1) * self.bs
            d = self.src.read(gdt + group * self.desc_size, self.desc_size)
            lo = struct.unpack_from("<I", d, 8)[0]
            hi = struct.unpack_from("<I", d, 0x28)[0] if self.desc_size >= 64 else 0
            self._gd[group] = (hi << 32) | lo
        return self._gd[group]

    def inode(self, ino):
        g, i = divmod(ino - 1, self.ipg)
        raw = self.src.read(self._inode_table(g) * self.bs + i * self.inode_size, self.inode_size)
        mode = struct.unpack_from("<H", raw, 0)[0]
        size = struct.unpack_from("<I", raw, 4)[0] | (struct.unpack_from("<I", raw, 0x6C)[0] << 32)
        flags = struct.unpack_from("<I", raw, 0x20)[0]
        return {"ino": ino, "mode": mode, "size": size, "flags": flags, "iblock": raw[0x28:0x28 + 60]}

    # --- data blocks -----------------------------------------------------------------------------------------
    def _extents(self, node):
        magic, entries, _mx, depth = struct.unpack_from("<4H", node, 0)
        if magic != 0xF30A:
            raise ValueError("bad extent header")
        for k in range(entries):
            e = node[12 + 12 * k: 24 + 12 * k]
            if depth == 0:
                lblk, ln, hi, lo = struct.unpack("<IHHI", e)
                init = ln <= 32768
                ln = ln if init else ln - 32768
                yield lblk, (hi << 32) | lo, ln, init
            else:
                lblk, lo, hi, _u = struct.unpack("<IIHH", e)
                yield from self._extents(self.block((hi << 32) | lo))

    def _blockmap(self, iblock, nblocks):
        ptrs = list(struct.unpack("<15I", iblock))
        out = list(ptrs[:12])

        def ind(b, level):
            if b == 0:
                return []
            arr = struct.unpack("<%dI" % (self.bs // 4), self.block(b))
            if level == 1:
                return list(arr)
            res = []
            for x in arr:
                res += ind(x, level - 1)
                if len(res) + len(out) >= nblocks:
                    break
            return res
        for lvl, p in zip((1, 2, 3), ptrs[12:15]):
            if len(out) >= nblocks:
                break
            out += ind(p, lvl)
        return out[:nblocks]

    def extent_list(self, ino_d):
        """[(logical block, physical block, length, initialized)] of a file's data, or None for a block-map file."""
        if not ino_d["flags"] & 0x80000:
            return None
        return list(self._extents(ino_d["iblock"]))

    def read_file(self, ino_d, limit=None):
        size = ino_d["size"] if limit is None else min(ino_d["size"], limit)
        if ino_d["flags"] & 0x10000000:
            raise NotImplementedError("inline_data")
        buf = bytearray(size)
        nblocks = (size + self.bs - 1) // self.bs
        if ino_d["flags"] & 0x80000:
            for lblk, pblk, ln, init in self._extents(ino_d["iblock"]):
                if lblk >= nblocks:
                    continue
                ln = min(ln, nblocks - lblk)
                if init:
                    data = self.block(pblk, ln)
                    s = lblk * self.bs
                    buf[s:s + len(data)] = data[: max(0, min(len(data), size - s))]
        else:
            for k, pb in enumerate(self._blockmap(ino_d["iblock"], nblocks)):
                if pb:
                    s = k * self.bs
                    data = self.block(pb)
                    buf[s:s + self.bs] = data[: max(0, min(self.bs, size - s))]
        return bytes(buf[:size])

    # --- directories / paths ---------------------------------------------------------------------------------
    def listdir(self, ino_d):
        data = self.read_file(ino_d)
        out = []
        for b in range(0, len(data), self.bs):
            blk = data[b:b + self.bs]
            p = 0
            while p + 8 <= len(blk):
                ino, rec_len, name_len, _ft = struct.unpack_from("<IHBB", blk, p)
                if rec_len < 8:
                    break
                if ino and name_len:
                    name = blk[p + 8:p + 8 + name_len].decode("utf-8", "replace")
                    if name not in (".", ".."):
                        out.append((name, ino))
                p += rec_len
        return out

    def readlink(self, ino_d):
        if ino_d["size"] < 60 and not (ino_d["flags"] & 0x80000):
            return ino_d["iblock"][: ino_d["size"]].decode("utf-8", "replace")
        return self.read_file(ino_d).decode("utf-8", "replace")

    def lookup(self, path, depth=0):
        if depth > 20:
            raise ValueError("symlink loop")
        cur = self.inode(2)
        parts = [p for p in path.split("/") if p]
        walked = []
        for i, part in enumerate(parts):
            if not stat.S_ISDIR(cur["mode"]):
                return None
            hit = dict(self.listdir(cur)).get(part)
            if hit is None:
                return None
            nxt = self.inode(hit)
            if stat.S_ISLNK(nxt["mode"]):
                tgt = self.readlink(nxt)
                base = "/" if tgt.startswith("/") else "/" + "/".join(walked)
                rest = "/".join(parts[i + 1:])
                return self.lookup(os.path.normpath(os.path.join(base, tgt, rest)).replace("\\", "/"), depth + 1)
            walked.append(part)
            cur = nxt
        return cur


def redact(data: bytes) -> bytes:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return data
    lines = [SECRET_RE.sub(r"\1<redacted>", ln) for ln in text.splitlines(keepends=True)]
    return "".join(lines).encode("utf-8")


def extract(fs, path, out_root, max_file, budget, log):
    node = fs.lookup(path)
    if node is None:
        log(f"MISSING  {path}")
        return budget
    stack = [(path.rstrip("/") or "/", node)]
    while stack and budget > 0:
        p, nd = stack.pop()
        if stat.S_ISDIR(nd["mode"]):
            for name, ino in sorted(fs.listdir(nd)):
                stack.append((p.rstrip("/") + "/" + name, fs.inode(ino)))
            continue
        dst = os.path.join(out_root, p.lstrip("/").replace("/", os.sep))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if stat.S_ISLNK(nd["mode"]):
            with open(dst + ".symlink.txt", "w", encoding="utf-8") as f:
                f.write(fs.readlink(nd))
            log(f"LINK     {p} -> {fs.readlink(nd)}")
            continue
        if not stat.S_ISREG(nd["mode"]):
            continue
        try:
            data = fs.read_file(nd, limit=max_file)
        except NotImplementedError:
            log(f"SKIP     {p} (inline_data)")
            continue
        data = redact(data)
        with open(dst, "wb") as f:
            f.write(data)
        budget -= len(data)
        log(f"FILE     {p} ({nd['size']} B{', truncated' if nd['size'] > max_file else ''})")
    return budget


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("source", help=r"\\.\PhysicalDriveN (admin) or an image file")
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--inspect", action="append", default=[], help="print a file's inode / extents (read-only)")
    ap.add_argument("--offset-mb", type=float, default=0.0, help="partition start in MiB")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-file-mb", type=float, default=64.0)
    ap.add_argument("--budget-mb", type=float, default=600.0)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    logf = open(os.path.join(a.out, "extract_log.txt"), "w", encoding="utf-8")

    def log(s):
        print(s)
        logf.write(s + "\n")
    try:
        fs = Ext4(Source(a.source, int(a.offset_mb * 1024 * 1024)))
    except (OSError, ValueError) as e:
        log(f"ERROR opening {a.source}: {e}")
        if isinstance(e, PermissionError) or "denied" in str(e).lower():
            log("Raw disks need an ADMIN prompt (run the terminal as administrator).")
        return 2
    log(f"ext4 volume '{fs.volume_name}', UUID {fs.uuid}, block size {fs.bs}, inode size {fs.inode_size}, "
        f"{fs.inodes_count} inodes, incompat 0x{fs.incompat:x}, ro_compat 0x{fs.ro_compat:x}")
    for p in a.inspect:
        nd = fs.lookup(p)
        if nd is None:
            log(f"INSPECT  {p}: missing")
            continue
        ext = fs.extent_list(nd)
        log(f"INSPECT  {p}: inode {nd['ino']} size {nd['size']} flags 0x{nd['flags']:x} extents {ext}")
        for lblk, pblk, ln, init in ext or []:
            log(f"INSPECT    logical {lblk} -> physical block {pblk} (+{ln}) {'initialized' if init else 'UNWRITTEN'}, "
                f"byte offset in partition {pblk * fs.bs}")
    budget = int(a.budget_mb * 1024 * 1024)
    for p in a.paths:
        budget = extract(fs, p, a.out, int(a.max_file_mb * 1024 * 1024), budget, log)
    log("done" if budget > 0 else "stopped: size budget reached")
    return 0


if __name__ == "__main__":
    sys.exit(main())

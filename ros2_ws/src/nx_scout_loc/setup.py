from setuptools import setup

package_name = "nx_scout_loc"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Ron Rastorguev",
    maintainer_email="ronrastorguev@gmail.com",
    description="Scout camera relocalization (ROS 2 adapter over localization/)",
    license="TODO",
    entry_points={"console_scripts": ["scout_loc = nx_scout_loc.loc_node:main"]},
)

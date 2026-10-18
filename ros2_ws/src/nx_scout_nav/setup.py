from setuptools import setup

package_name = "nx_scout_nav"

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
    description="Scout autonomous flight: takeoff, A to a wall point, looming avoidance (ROS 2 adapter over navigation/)",
    license="TODO",
    entry_points={"console_scripts": ["scout_mission = nx_scout_nav.mission_node:main"]},
)

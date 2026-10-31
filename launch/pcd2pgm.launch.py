import os
import launch
import launch_ros

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, LogInfo, OpaqueFunction, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_prefix, get_package_share_directory

def launch_setup(context):

    pcd2pgm_config = LaunchConfiguration('pcd2pgm_config').perform(context)
    use_gui = LaunchConfiguration('gui').perform(context).lower() in ('true', '1')

    parameters = [pcd2pgm_config]
    if use_gui:
        # pcd2pgm_gui.py は save_all_cloud で保存される中間点群を読む
        parameters.append({'save_all_cloud': True})

    pcd2pgm_node = Node(
        package='pcd2pgm',
        executable='pcd2pgm_node',
        name='pcd2pgm_node',
        output='screen',
        parameters=parameters
    )

    actions = [pcd2pgm_node]

    if use_gui:
        gui = ExecuteProcess(
            cmd=[os.path.join(get_package_prefix('pcd2pgm'), 'lib', 'pcd2pgm', 'pcd2pgm_gui.py'),
                 '--config', pcd2pgm_config],
            output='screen'
        )

        def on_node_exit(event, context):
            if event.returncode != 0:
                return [LogInfo(msg='pcd2pgm_node failed, GUI is not started.')]
            return [gui]

        actions.append(RegisterEventHandler(OnProcessExit(target_action=pcd2pgm_node, on_exit=on_node_exit)))

    return actions

def generate_launch_description():

    ld = LaunchDescription()

    ld.add_action(DeclareLaunchArgument(
        'pcd2pgm_config',
        default_value=os.path.join(
            get_package_share_directory('pcd2pgm'),
            'config',
            'config_pcd2pgm.yaml'
        )
    ))
    # true: pcd2pgm_node の終了後に pcd2pgm_gui.py を起動する
    ld.add_action(DeclareLaunchArgument('gui', default_value='false'))

    ld.add_action(OpaqueFunction(function=launch_setup))

    return ld

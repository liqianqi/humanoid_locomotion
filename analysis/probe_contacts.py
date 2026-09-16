"""Small read-only diagnostic rollout; does not modify training configs or checkpoints."""
import argparse,json,sys
from pathlib import Path
from isaaclab.app import AppLauncher
parser=argparse.ArgumentParser()
parser.add_argument('--self-collisions',action='store_true')
parser.add_argument('--historical-gains',action='store_true')
parser.add_argument('--steps',type=int,default=50)
parser.add_argument('--output',required=True)
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
app=AppLauncher(args).app
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from humanoid_locomotion.velocity_env_cfg import RsxFlatEnvCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.mdp.terminations import illegal_contact
cfg=RsxFlatEnvCfg()
cfg.scene.num_envs=4
cfg.scene.robot.spawn.articulation_props.enabled_self_collisions=args.self_collisions
cfg.scene.robot.spawn.articulation_props.solver_position_iteration_count=8
cfg.scene.terrain.visual_material=None
cfg.scene.sky_light=None
cfg.commands.base_velocity.debug_vis=False
cfg.observations.policy.enable_corruption=False
cfg.terminations.base_contact.params['threshold']=1.0
cfg.events.reset_base.params['pose_range']={}
if args.historical_gains:
 cfg.scene.robot.actuators['rs06'].stiffness=200.;cfg.scene.robot.actuators['rs06'].damping=5.
 cfg.scene.robot.actuators['rs02'].stiffness=180.;cfg.scene.robot.actuators['rs02'].damping=4.
records=[]
def record_contact(env,threshold,sensor_cfg):
 sensor=env.scene['contact_forces'];robot=env.scene['robot']
 norms=sensor.data.net_forces_w_history.norm(dim=-1).max(dim=1).values
 done=illegal_contact(env,threshold,sensor_cfg)
 records.append({'step':len(records),'base_z':robot.data.root_pos_w[:,2].tolist(),'base_force':norms[:,sensor.body_names.index('base_link')].tolist(),'done':done.tolist(),'forces_env0':dict(zip(sensor.body_names,norms[0].tolist())),'q_env0':dict(zip(robot.joint_names,robot.data.joint_pos[0].tolist()))})
 return done
cfg.terminations.base_contact.func=record_contact
env=ManagerBasedRLEnv(cfg=cfg)
env.reset()
print('PROBE_JOINTS',env.scene['robot'].joint_names,flush=True)
for _ in range(args.steps):
 with torch.inference_mode():env.step(torch.zeros((4,env.action_manager.total_action_dim),device=env.device))
Path(args.output).write_text(json.dumps({'args':vars(args),'records':records},indent=2))
print('PROBE_RESULT',json.dumps({'self_collisions':args.self_collisions,'historical_gains':args.historical_gains,'done_count':sum(sum(r['done']) for r in records),'first':records[:3],'last':records[-1]}),flush=True)
env.close();app.close()

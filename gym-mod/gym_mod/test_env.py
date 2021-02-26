import gym
import moviepy.editor as mpy

from method.environments import make_env
from config import create_s2r_parser


parser = create_s2r_parser()
args, _ = parser.parse_known_args()
env = make_env("SawyerPush-v0", args)


cnt_episodes = 0

env.reset()
if args.record_video:
    frames = [env.render("rgb_array")]

while True:
    _, _, done, _ = env.step(env.action_space.sample())

    if args.record_video:
        frames.append(env.render("rgb_array"))
    else:
        env.render()

    if done:
        path = "SawyerPush-v0-%d.mp4" % cnt_episodes
        fps = 15.0
        def f(t):
            frame_length = len(frames)
            new_fps = 1.0 / (1.0 / fps + 1.0 / frame_length)
            idx = min(int(t * new_fps), frame_length - 1)
            return frames[idx]

        video = mpy.VideoClip(f, duration=len(frames) / fps + 2)
        video.write_videofile(path, fps, verbose=False)
        print("[*] save video to %s" % path)
        cnt_episodes += 1
        env.reset()
        frames = [env.render("rgb_array")]


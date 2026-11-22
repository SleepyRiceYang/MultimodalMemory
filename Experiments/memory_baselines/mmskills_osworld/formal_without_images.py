"""Run the frozen 40 tasks with state cards but no skill reference images."""
import sys
import formal_three as monitoring

if __name__ == '__main__':
    monitoring.CONDITIONS = ('visual_skill_without_images',)
    if '--output' not in sys.argv:
        sys.argv.extend(['--output', str(monitoring.ROOT / 'Experiments/memory_baselines/artifacts/mmskills_osworld/formal_40_without_images_20261111_v1')])
    monitoring.main()

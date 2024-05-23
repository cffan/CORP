import argparse

from falcon_challenge.config import FalconConfig, FalconTask
from falcon_challenge.evaluator import FalconEvaluator

from corp_decoder import CORPDecoder

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluation", type=str, required=True, choices=["local", "remote"]
    )
    parser.add_argument(
        "--corp-config", type=str, default='config/falcon_held_out_eval.yaml',
        help="Path to CORP config file."
    )
    parser.add_argument(
        '--split', type=str, choices=['h1', 'h2', 'm1', 'm2'], default='h2',
    )
    parser.add_argument(
        '--phase', choices=['minival', 'test'], default='minival'
    )
    args = parser.parse_args()

    evaluator = FalconEvaluator(
        eval_remote=args.evaluation == "remote",
        split=args.split)

    task = getattr(FalconTask, args.split)
    config = FalconConfig(task=task)

    decoder = CORPDecoder(
        task_config=config,
        corp_config_path=args.corp_config,
    )

    evaluator.evaluate(decoder, phase=args.phase)
    # print(evaluator.evaluate_files(decoder, evaluator.get_eval_files(phase=args.phase)[:]))


if __name__ == "__main__":
    main()
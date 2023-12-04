import argparse
import json
import os
import random
from collections import defaultdict
from typing import Callable

import more_itertools
import numpy as np
import torch
from coco_metric import compute_cider, postprocess_captioning_generation
from eval_datasets import COCOTrainDataset, COCOTestDataset
from tqdm import tqdm

from open_flamingo.src.factory import create_model_and_transforms
import copy

parser = argparse.ArgumentParser()
parser.add_argument("--lm_path", type=str, default="facebook/opt-1.3b")
parser.add_argument("--lm_tokenizer_path", type=str, default="facebook/opt-30b")
parser.add_argument("--vision_encoder_path", default="ViT-L-14", type=str)
parser.add_argument("--vision_encoder_pretrained", default="openai", type=str)
parser.add_argument("--checkpoint_path", type=str, required=True)
parser.add_argument(
    "--cross_attn_every_n_layers",
    type=int,
    default=1,
    help="how often to add a cross-attention layer after each transformer layer",
)
parser.add_argument(
    "--results_file", type=str, default=None, help="JSON file to save results"
)

# Trial arguments
parser.add_argument("--shots", nargs="+", default=[0, 4, 8, 16, 32], type=int)
parser.add_argument(
    "--num_trials",
    type=int,
    default=1,
    help="Number of trials to run for each shot using different demonstrations",
)
parser.add_argument(
    "--trial_seeds",
    nargs="+",
    default=[0],
    help="Seeds to use for each trial for picking demonstrations and eval sets",
)
parser.add_argument(
    "--seed",
    default=0,
    help="Seeds to use for each trial for picking demonstrations and eval sets",
)
parser.add_argument(
    "--num_samples", type=int, default=5000, help="Number of samples to evaluate on"
)

parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--device", type=int, default=0)

## COCO Dataset
parser.add_argument(
    "--coco_image_dir_path",
    type=str,
    help="Path to the flickr30/flickr30k_images directory.",
    default=None,
)
parser.add_argument(
    "--coco_annotations_json_path",
    type=str,
    default=None,
)

parser.add_argument(
    "--mgc_path",
    type=str,
    default=None,
)

# Per-dataset evaluation flags
parser.add_argument(
    "--mgc",
    action="store_true",
    default=False,
    help="Whether to evaluate on MGC.",
)

parser.add_argument(
    "--mgca_path",
    type=str,
    default=None,
)

parser.add_argument(
    "--mgca",
    action="store_true",
    default=False,
    help="Whether to evaluate on MGCA.",
)

parser.add_argument(
    "--clip",
    action="store_true",
    default=False,
    help="Whether to evaluate on SIIR.",
)

def main():
    args = parser.parse_args()

    device = "cuda:" + str(args.device)

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    torch.cuda.manual_seed(int(args.seed))

    precision = 'fp16'

    # load model
    flamingo, image_processor, tokenizer = create_model_and_transforms(
        clip_vision_encoder_path="ViT-L-14",
        clip_vision_encoder_pretrained="openai",
        lang_encoder_path=args.lm_path,
        tokenizer_path=args.lm_tokenizer_path,
        cross_attn_every_n_layers=4,
        # new params
        inference=True,
        precision=precision,
        device=device,
        checkpoint_path=args.checkpoint_path,
    )

    results = defaultdict(list)


    print("Evaluating on COCO...")
    for shot in args.shots:
        scores = []
        clipscores = []
        RefCLIPScores = []
        for seed, trial in zip(args.trial_seeds, range(args.num_trials)):
            cider_score, clipscore, RefCLIPScore = evaluate_coco_flickr(
                model=flamingo,
                tokenizer=tokenizer,
                image_processor=image_processor,
                batch_size=args.batch_size,
                image_dir_path=args.coco_image_dir_path,
                annotations_json_path=args.coco_annotations_json_path,
                mgc_path=args.mgc_path,
                mgca_path=args.mgca_path,
                mgc=args.mgc,
                mgca=args.mgca,
                clip=args.clip,
                num_samples=args.num_samples,
                num_shots=shot,
                num_beams=3,
                precision=precision,
                results_file=args.results_file,
                device=device,
                seed=seed,
            )
            print(f"Shots {shot} Trial {trial} CIDEr score: {cider_score}, CLIP score: {clipscore}, RefCLIPScore: {RefCLIPScore}")
            scores.append(cider_score)
            clipscores.append(clipscore)
            RefCLIPScores.append(RefCLIPScore)
        print(f"Shots {shot} Mean CIDEr score: {np.mean(scores)}, CLIP score: {np.mean(clipscores)}, RefCLIPScore: {np.mean(RefCLIPScores)}")
        results["coco"].append(
            {"shots": shot, "trials": scores, "mean": np.mean(scores), "clipscore": np.mean(clipscores), "RefCLIPScore": np.mean(RefCLIPScores)}
        )

    if args.results_file is not None:
        with open(args.results_file, "w") as f:
            json.dump(results, f, indent=4)


def get_random_indices(num_samples, query_set_size, full_dataset, seed):
    if num_samples + query_set_size > len(full_dataset):
        raise ValueError(
            f"num_samples + num_shots must be less than {len(full_dataset)}"
        )
    
    random_indices = np.random.choice(
        len(full_dataset), num_samples + query_set_size, replace=False
    )
    return random_indices


def prepare_eval_samples_and_dataset(full_dataset, random_indices, query_set_size):
    # get in context samples
    in_context_samples = [full_dataset[i] for i in random_indices[:query_set_size]]
    eval_dataset = torch.utils.data.Subset(
        full_dataset, random_indices[query_set_size:]
    )
    return in_context_samples, eval_dataset


def get_context_images(image_processor, in_context_samples, num_shots):
    if num_shots > 0:
        context_images = [
            image_processor(s["image"]).unsqueeze(0) for s in in_context_samples
        ]
        context_images = torch.cat(context_images, dim=0)
        context_images = context_images.unsqueeze(1).unsqueeze(0)
    else:
        context_images = None
    return context_images


def get_context_text(
    get_prompt: Callable[[dict], str],
    in_context_samples,
    effective_num_shots,
    num_shots,
    mgca,
) -> str:
    if effective_num_shots>0:
        context_text = (
            "".join([get_prompt(s, s['WC_gt_idx']) for s in in_context_samples])
            if mgca
            else "".join([get_prompt(s, 0) for s in in_context_samples])
        )
    else:
        context_text = ""
    if num_shots == 0:
        context_text = context_text.replace("<image>", "")
    return context_text


def prepare_batch_images(batch, image_processor, context_images, num_shots):
    batch_images = None
    for b, sample_imgs in zip(batch, context_images):
        b_image = image_processor(b["image"]).unsqueeze(0).unsqueeze(1).unsqueeze(0)
        b_image = torch.cat([sample_imgs, b_image], dim=1) if num_shots > 0 else b_image

        if batch_images is None:
            batch_images = b_image
        else:
            batch_images = torch.cat([batch_images, b_image], dim=0)
    return batch_images


def sample_batch_demos_from_query_set(query_set, num_samples, batch, clip = False):
    if not clip:
        return [[query_set[i] for i in random.sample(range(len(query_set)), num_samples)] for _ in range(len(batch))]
    else:
        output = []
        for i in range(len(batch)):
            o = []
            for id in batch[i]["clip_image_ids"][:num_samples]:
                x = copy.deepcopy(query_set.id2item(id))
                o.append(x)
            output.append(o)
        return output

def get_outputs(
    model,
    batch_images,
    device,
    attention_mask,
    max_generation_length,
    num_beams,
    length_penalty,
    input_ids,
):
    
    with torch.inference_mode():
        outputs = model.generate(
            batch_images.to(device),
            input_ids.to(device),
            attention_mask=attention_mask.to(device),
            max_new_tokens=max_generation_length,
            num_beams=num_beams,
            length_penalty=length_penalty,
        )

    outputs = outputs[:, len(input_ids[0]) :]
    return outputs


def evaluate_coco_flickr(
    model,
    tokenizer,
    image_processor,
    batch_size,
    image_dir_path,
    annotations_json_path,
    mgc_path,
    mgca_path,
    mgc,
    mgca,
    clip,
    seed=42,
    max_generation_length=20,
    num_beams=3,
    length_penalty=-2.0,
    num_samples=5000,
    query_set_size=2048,
    num_shots=8,
    precision="fp32",
    results_file="results_baseline.json",
    device="cpu",
):
    """Evaluate a model on COCO dataset.

    Args:
        model (nn.Module): model to evaluate
        tokenizer (transformers.PreTrainedTokenizer): tokenizer for the model
        image_processor : image processor for the model
        batch_size (int): batch size
        image_dir_path (str, optional): path to the directory containing the images.
        annotations_json_path (str, optional): path to the json file containing the annotations.
        seed (int, optional): seed for random number generator. Defaults to 42.
        max_generation_length (int, optional): maximum length of the generated caption. Defaults to 10.
        num_beams (int, optional): number of beams to use for beam search. Defaults to 3.
        length_penalty (float, optional): length penalty for beam search. Defaults to -2.0.
        num_samples (int, optional): number of samples to evaluate on. Defaults to 5000.
        query_set_size (int, optional): number of samples to use for query set. Defaults to 2048.
        num_shots (int, optional): number of in-context samples to use. Defaults to 8.
        device (int, optional): device to use. Defaults to -1.
        num_workers (int, optional): number of workers to use for dataloader. Defaults to 4.
        is_flickr (bool): defines if that data is COCO or Flickr. Defaults to False (COCO).

    Returns:
        float: CIDEr score

    """

    full_dataset = COCOTrainDataset(
        image_dir_path=image_dir_path,
        annotations_path=annotations_json_path,
        WC_captions_path=mgc_path,
        WC_best_gt_path=mgca_path,
    )
    effective_num_shots = num_shots if num_shots > 0 else 2

    eval_dataset = COCOTestDataset()
    model.eval()

    if mgc and not mgca:
        prompt_cap = 'WC_captions'
    else:
        prompt_cap = 'captions'

    def get_prompt(sample, idx = 0):
        return f"<image>Output:{sample[prompt_cap][idx].strip()}<|endofchunk|>"

    predictions = defaultdict()

    desc = "Running inference COCO"

    for batch in more_itertools.chunked(tqdm(eval_dataset, desc=desc), batch_size):
        batch_demo_samples = sample_batch_demos_from_query_set(
            full_dataset, effective_num_shots, batch, clip
        )

        context_images = [
            get_context_images(
                image_processor=image_processor,
                in_context_samples=batch_demo_samples[i],
                num_shots=num_shots,
            )
            for i in range(len(batch))
        ]

        context_text = [
            get_context_text(
                get_prompt,
                in_context_samples=batch_demo_samples[i],
                effective_num_shots=effective_num_shots,
                num_shots=num_shots,
                mgca=mgca,
            )
            for i in range(len(batch))
        ]

        batch_images = prepare_batch_images(
            batch=batch,
            image_processor=image_processor,
            context_images=context_images,
            num_shots=num_shots,
        )

        batch_text = [f"{context_text[i]}<image>Output:" for i in range(len(batch))]

        tokenizer.padding_side = "left"
        encodings = tokenizer(
            batch_text,
            padding="longest",
            truncation=True,
            return_tensors="pt",
            max_length=2000,
        )
        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]

        outputs = get_outputs(
            model=model,
            batch_images=batch_images.half() if precision == 'fp16' else batch_images,
            device=device,
            attention_mask=attention_mask,
            max_generation_length=max_generation_length,
            num_beams=num_beams,
            length_penalty=length_penalty,
            input_ids=input_ids,
        )
        new_predictions = [
            postprocess_captioning_generation(out).replace('"', "")
            for out in tokenizer.batch_decode(outputs, skip_special_tokens=True)
        ]

        for i, sample in enumerate(batch):
            predictions[sample["image_id"]] = {
                "caption": new_predictions[i],
                "prompt_text": batch_text[i],
                "prompt_images": [img['image_id'] for img in batch_demo_samples[i]],
            }

    if not os.path.exists(results_file.split(".")[0]):
        os.mkdir(results_file.split(".")[0])
    random_uuid = os.path.join(results_file.split(".")[0], "{}_{}".format(results_file.split(".")[0], num_shots))
    results_path = f"{random_uuid}.json"
    with open(results_path, "w") as f:
        f.write(
            json.dumps(
                [
                    {"image_id": k, "caption": predictions[k]["caption"], "prompt_text": predictions[k]["prompt_text"],
                        "prompt_images": predictions[k]["prompt_images"]}
                    for k in predictions
                ],
                indent=4,
            )
        )

    metrics = compute_cider(
        result_path=results_path,
        annotations_path=eval_dataset.annotations_path,
    )

    return metrics["CIDEr"] * 100.0


if __name__ == "__main__":
    main()

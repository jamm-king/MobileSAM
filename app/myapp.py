import gradio as gr
import numpy as np
import torch
from mobile_sam import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
from PIL import ImageDraw, Image
from utils.tools import box_prompt, format_results, point_prompt

# Most of our demo code is from [FastSAM Demo](https://huggingface.co/spaces/An-619/FastSAM). Huge thanks for AN-619.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pre-trained model
sam_checkpoint = "../weights/mobile_sam.pt"
model_type = "vit_t"

mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
mobile_sam = mobile_sam.to(device=device)
mobile_sam.eval()

mask_generator = SamAutomaticMaskGenerator(mobile_sam)
predictor = SamPredictor(mobile_sam)

examples = [
    ["assets/picture3.jpg"],
    ["assets/picture4.jpg"],
    ["assets/picture5.jpg"],
    ["assets/picture6.jpg"],
    ["assets/picture1.jpg"],
    ["assets/picture2.jpg"],
]

default_example = examples[0]

global_points = []
global_point_label = []


@torch.no_grad()
def segment_everything(
    image,
    input_size=1024,
):
    global mask_generator

    input_size = int(input_size)
    w, h = image.size
    scale = input_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    image = image.resize((new_w, new_h))

    nd_image = np.array(image)
    annotations = mask_generator.generate(nd_image)

    return annotations


def segment_with_points(
    image,
    input_size=1024,
):
    global global_points
    global global_point_label

    input_size = int(input_size)
    w, h = image.size
    scale = input_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    image = image.resize((new_w, new_h))

    scaled_points = np.array(
        [[int(x * scale) for x in point] for point in global_points]
    )
    scaled_point_label = np.array(global_point_label)

    if scaled_points.size == 0 and scaled_point_label.size == 0:
        print("No points selected")
        return image, image

    print(scaled_points, scaled_points is not None)
    print(scaled_point_label, scaled_point_label is not None)

    nd_image = np.array(image)
    predictor.set_image(nd_image)
    masks, scores, logits = predictor.predict(
        point_coords=scaled_points,
        point_labels=scaled_point_label,
        multimask_output=True,
    )

    results = format_results(masks, scores, logits, 0)

    annotations, _ = point_prompt(
        results, scaled_points, scaled_point_label, new_h, new_w
    )
    annotations = np.array([annotations])

    global_points = []
    global_point_label = []

    return annotations


image = Image.open(default_example[0])
annotations = segment_everything(image)

print("annotations-length:", len(annotations))

for annotation in annotations:
    print("shape:", annotation['segmentation'].shape)
    print("point_coords:", annotation['point_coords'])
    print("stability_score:", annotation['stability_score'])

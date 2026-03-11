import sys
import torch
import onnxruntime as ort
from PIL import Image
from torchvision import transforms


def preprocess(image_path):

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5,0.5,0.5],
            std=[0.5,0.5,0.5]
        )
    ])

    image = Image.open(image_path).convert("RGB")

    tensor = transform(image).unsqueeze(0)

    return tensor.numpy()


def main():

    model_path = "imagenet_model.onnx"

    image_path = sys.argv[1]

    session = ort.InferenceSession(model_path)

    input_tensor = preprocess(image_path)

    outputs = session.run(
        None,
        {"input": input_tensor}
    )

    pred = outputs[0].argmax()

    print("Predicted class:", pred)


if __name__ == "__main__":
    main()

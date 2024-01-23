from segments import SegmentsClient
from segments.huggingface import release2dataset
from segments.utils import get_semantic_bitmap

# Get a specific dataset release
client = SegmentsClient("Your Segments.ai API Key")  # Segments.ai API Key
release = client.get_release("your_Segments.ai_id/your_Segments.ai_dataset", "v0.1")  # Segments.ai dataset name and version
hf_dataset = release2dataset(release)

def convert_segmentation_bitmap(example):
    return {
        "label.segmentation_bitmap":
            get_semantic_bitmap(
                example["label.segmentation_bitmap"],
                example["label.annotations"],
                id_increment=0,
            )
    }

semantic_dataset = hf_dataset.map(
    convert_segmentation_bitmap,
)

semantic_dataset = semantic_dataset.rename_column('image', 'pixel_values')
semantic_dataset = semantic_dataset.rename_column('label.segmentation_bitmap', 'label')
semantic_dataset = semantic_dataset.remove_columns(['name', 'uuid', 'status', 'label.annotations'])

semantic_dataset.push_to_hub("your_hf_id/your_hf_dataset") # The name of a HF user/dataset


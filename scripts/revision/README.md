To compile words representative of each feature category, we utilized XX.

```
python video_annotator.py --input /Users/h/Documents/projects_local/life-encoding/scripts/revision/videos/ses-01_run-01_order-02_content-wanderers.mp4 --output-dir /Users/h/Documents/projects_local/life-encoding/scripts/revision/annotation_output
```

Update list of words in
scripts/revision/annotation/video_annotation_system/models/annotation_generator.py

python /Users/h/Documents/projects_local/life-encoding/scripts/revision/annotation/video_annotation_system/video_annotator.py --input '/Users/h/Documents/projects_local/life-encoding/scripts/revision/videos/ses-01_run-01_order-02_content-wanderers.mp4' --output-dir /Users/h/Documents/projects_local/life-encoding/scripts/revision/annotation_output/

python /Users/h/Documents/projects_local/life-encoding/scripts/revision/annotation/video_annotation_system/video_annotator.py --input /Users/h/Documents/projects_local/life-encoding/scripts/revision/videos/ses-01_run-01_order-02_content-wanderers.mp4 \
 --action-model openai/clip-vit-base-patch32 \
 --agent-model openai/clip-vit-base-patch32 \
 --scene-model openai/clip-vit-base-patch32 \
 --object-model openai/clip-vit-base-patch32 \
 --output-dir /Users/h/Documents/projects_local/life-encoding/scripts/revision/annotation_output/

python /Users/h/Documents/projects_local/life-encoding/scripts/revision/annotation/video_annotation_system/video_annotator.py --input /Users/h/Documents/projects_local/life-encoding/scripts/revision/videos/ses-01_run-01_order-02_content-wanderers.mp4 \
 --action-model facebook/slowfast \
 --agent-model google/vit-base-patch16-224 \
 --scene-model openai/clip-vit-base-patch32 \
 --object-model openai/clip-vit-base-patch32 \
 --output-dir /Users/h/Documents/projects_local/life-encoding/scripts/revision/annotation_output_2/

    if model_type.lower() == "clip":
        model_name = model_name or "openai/clip-vit-base-patch32"
        model_dict = load_clip_model(model_name, device)
    elif model_type.lower() == "vit":
        model_name = model_name or "google/vit-base-patch16-224"
        model_dict = load_vit_model(model_name, device)
    elif model_type.lower() == "slowfast":
        model_name = model_name or "facebook/slowfast"
        model_dict = load_slowfast_model(model_name, device)

    python video_annotator.py --input-folder /videos/ --action-model facebook/slowfast --agent-model openai/clip --scene-model google/vit

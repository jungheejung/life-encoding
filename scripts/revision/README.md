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

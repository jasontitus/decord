# Creative Commons Test Videos

These test videos are generated using ffmpeg with properties inspired by well-known
Creative Commons licensed films from the Blender Foundation.

## Generated Test Videos

The following videos were generated locally using ffmpeg's built-in test sources
(testsrc2, testsrc, smptebars) to create deterministic, reproducible test data
with known properties:

| File | Resolution | FPS | Duration | Frames | Source Pattern |
|------|-----------|-----|----------|--------|---------------|
| `bbb_style_640x360_24fps.mp4` | 640x360 | 24 | 5s | 120 | testsrc2 |
| `sintel_style_320x240_30fps.mp4` | 320x240 | 30 | 3s | 90 | testsrc |
| `tos_style_480x270_25fps.mp4` | 480x270 | 25 | 8s | 200 | smptebars |

All videos are H.264 encoded with yuv420p pixel format.

## Creative Commons Video Sources

The video properties are inspired by (and tests are designed to validate decord
against properties matching) these Creative Commons licensed films:

### Big Buck Bunny
- **License:** CC-BY-3.0
- **Copyright:** (c) 2008 Blender Foundation | www.bigbuckbunny.org
- **Download:** https://test-videos.co.uk/bigbuckbunny/mp4-h264
- **Alternative:** http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4
- **Project:** https://peach.blender.org/

### Sintel
- **License:** CC-BY-3.0
- **Copyright:** (c) 2010 Blender Foundation | durian.blender.org
- **Download:** https://test-videos.co.uk/vids/sintel/mp4/h264/360/Sintel_360_10s_1MB.mp4
- **Alternative:** http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/Sintel.mp4
- **Project:** https://durian.blender.org/

### Tears of Steel
- **License:** CC-BY-3.0
- **Copyright:** (c) 2012 Blender Foundation | mango.blender.org
- **Download:** https://test-videos.co.uk/vids/tearsofsteel/mp4/h264/360/Tears_of_Steel_360_10s_1MB.mp4
- **Alternative:** http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/TearsOfSteel.mp4
- **Project:** https://mango.blender.org/

## Regenerating Test Videos

To regenerate these test videos, run:

```bash
ffmpeg -y -f lavfi -i "testsrc2=duration=5:size=640x360:rate=24" \
  -pix_fmt yuv420p -c:v libx264 -preset medium -crf 23 \
  bbb_style_640x360_24fps.mp4

ffmpeg -y -f lavfi -i "testsrc=duration=3:size=320x240:rate=30" \
  -pix_fmt yuv420p -c:v libx264 -preset medium -crf 28 \
  sintel_style_320x240_30fps.mp4

ffmpeg -y -f lavfi -i "smptebars=duration=8:size=480x270:rate=25" \
  -pix_fmt yuv420p -c:v libx264 -preset medium -crf 23 \
  tos_style_480x270_25fps.mp4
```

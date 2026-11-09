from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
from PIL import Image, ImageDraw, ImageFont
import os

root = Path(__file__).resolve().parent.parent
build = root / ".diapredict_build"
source = root / "deliverables" / "DiaPredict_EHR_Analytics_v16_next_visit_forecasts.pptx"
replacement = build / "next_visit_diagram.png"
temporary = build / "v16_repacked.pptx"

scale = 2
img = Image.new("RGB", (1330, 584), "#f2f6fc")
draw = ImageDraw.Draw(img)
font_path = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "segoeui.ttf"
bold_path = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "segoeuib.ttf"

def font(size, bold=False):
    return ImageFont.truetype(str(bold_path if bold else font_path), size * scale)

def box(xy, fill, radius=14, outline=None, width=1):
    xy = tuple(int(v * scale) for v in xy)
    draw.rounded_rectangle(xy, radius=radius * scale, fill=fill, outline=outline,
                           width=width * scale if outline else 1)

def text(x, y, value, size=14, fill="#1b2b40", bold=False, anchor=None):
    draw.text((x * scale, y * scale), value, font=font(size, bold), fill=fill, anchor=anchor)

box((12, 10, 653, 282), "#ffffff", 20, "#d5deea", 1)
box((24, 17, 641, 54), "#0b2545", 12)
text(332, 35, "ПОВНИЙ ПРОФІЛЬ · 6 ВИМІРЯНИХ ПОКАЗНИКІВ", 14, "#ffffff", True, "mm")

labels = ["Вік", "Глюкоза натще", "HbA1c", "ІМТ", "Систолічний АТ", "Тригліцериди"]
coords = [(28, 64), (232, 64), (436, 64), (28, 108), (232, 108), (436, 108)]
for label, (x, y) in zip(labels, coords):
    box((x, y, x + 190, y + 38), "#edf3fd", 10)
    text(x + 95, y + 19, label, 13, "#203e75", True, "mm")

box((64, 155, 616, 219), "#e4efff", 15, "#b6d0ff", 1)
text(340, 173, "ОЧІКУВАНА ДАТА НАСТУПНОГО ВІЗИТУ", 12, "#294782", True, "mm")
text(340, 194, "Останній повний візит + медіана ≥2 інтервалів", 12, "#1b2b40", True, "mm")
text(340, 210, "Менше 3 повних візитів або пропуск → прогноз недоступний", 9, "#526783", False, "mm")

box((34, 228, 321, 273), "#203e75", 12)
text(177, 242, "TIER 1", 11, "#bcd8ff", True, "mm")
text(177, 259, "Імовірність першого запису E11", 10, "#ffffff", True, "mm")
box((341, 228, 628, 273), "#ffffff", 12, "#cdd9ee", 1)
text(484, 242, "TIER 2", 11, "#2770dd", True, "mm")
text(484, 259, "Прогноз HbA1c на ту саму дату", 10, "#1b2b40", True, "mm")
img.resize((665, 292), Image.Resampling.LANCZOS).save(replacement)

with ZipFile(source, "r") as zin, ZipFile(temporary, "w", ZIP_DEFLATED) as zout:
    replaced = False
    for item in zin.infolist():
        data = replacement.read_bytes() if item.filename == "ppt/media/image3.png" else zin.read(item.filename)
        if item.filename == "ppt/media/image3.png":
            replaced = True
        zout.writestr(item, data)
if not replaced:
    temporary.unlink(missing_ok=True)
    raise RuntimeError("Expected slide diagram image was not found in the presentation.")
temporary.replace(source)
print("Replaced the outdated screenshot with the complete-case forecast flow diagram.")

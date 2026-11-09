import fs from 'node:fs/promises';
import { FileBlob, PresentationFile } from 'file:///C:/Users/Admin/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs';

const root = 'C:/Users/Admin/OneDrive/Рабочий стол/Dowload_datasets';
const path = `${root}/deliverables/DiaPredict_EHR_Analytics_v16_next_visit_forecasts.pptx`;
const p = await PresentationFile.importPptx(await FileBlob.load(path));
for (const index of [2, 5, 6, 7, 8, 10]) {
  const png = await p.export({ slide: p.slides.items[index], format: 'png', scale: 1 });
  await fs.writeFile(`${root}/.diapredict_build/v16-verified-slide-${String(index + 1).padStart(2, '0')}.png`, new Uint8Array(await png.arrayBuffer()));
}
const report = await p.inspect({ kind: 'slide,textbox,notes', maxChars: 100000 });
await fs.writeFile(`${root}/.diapredict_build/v16-verified.inspect.ndjson`, report.ndjson, 'utf8');
console.log(`Reopened and rendered ${p.slides.items.length} slides.`);

import { FileBlob, PresentationFile } from 'file:///C:/Users/Admin/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs';
const p = await PresentationFile.importPptx(await FileBlob.load('C:/Users/Admin/OneDrive/Рабочий стол/Dowload_datasets/deliverables/DiaPredict_EHR_Analytics_v14_data_honest.pptx'));
for (let i=0;i<p.slides.items.length;i++) {
  const s=p.slides.items[i];
  if (![2,4,5,6,7,8,10].includes(i)) continue;
  console.log('SLIDE',i+1);
  for (const shape of s.shapes.items) console.log(shape.name, JSON.stringify(shape.text?.toString?.() ?? ''));
}
const r = await p.inspect({kind:'notes',maxChars:30000});
for (const line of r.ndjson.split('\n')) if (line.includes('"slideIndex":5') || line.includes('"slideIndex":6') || line.includes('"slideIndex":8') || line.includes('"slideIndex":10')) console.log(line);

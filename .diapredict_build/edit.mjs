import fs from 'node:fs/promises';
import { FileBlob, PresentationFile } from 'file:///C:/Users/Admin/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs';
const input='C:/Users/Admin/Downloads/DiaPredict_EHR_Analytics_v9.pptx';
const p=await PresentationFile.importPptx(await FileBlob.load(input));
const old='Код проєкту: github.com/NexoreSystems/DiaPredict';
const updated='Результат системи підтримує рішення лікаря й не є діагнозом.';
const report=await p.inspect({kind:'textbox',search:old,maxChars:3000});
const line=report.ndjson.split('\n').map(x=>JSON.parse(x)).find(x=>x.kind==='textbox'&&x.text===old);
if(!line) throw new Error('Target conclusion line not found; deck unchanged.');
p.resolve(line.id).text.replace(old,updated);
const s=p.slides.items[10];
for(let i=0;i<p.slides.items.length;i++){
 const png=await p.export({slide:p.slides.items[i],format:'png',scale:1});
 await fs.writeFile(`C:/Users/Admin/OneDrive/Рабочий стол/Dowload_datasets/.diapredict_build/final-slide-${String(i+1).padStart(2,'0')}.png`,new Uint8Array(await png.arrayBuffer()));
}
const pptx=await PresentationFile.exportPptx(p);
await pptx.save('C:/Users/Admin/OneDrive/Рабочий стол/Dowload_datasets/.diapredict_build/candidate.pptx');
console.log((await p.inspect({kind:'textbox',search:updated,maxChars:1000})).ndjson);

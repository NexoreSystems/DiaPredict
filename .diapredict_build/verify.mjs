import {FileBlob,PresentationFile} from 'file:///C:/Users/Admin/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs';
const p=await PresentationFile.importPptx(await FileBlob.load('C:/Users/Admin/OneDrive/Рабочий стол/Dowload_datasets/deliverables/DiaPredict_EHR_Analytics_v11_reviewed.pptx'));
console.log('slides',p.slides.items.length);
for (const q of ['не прогнозує','не є каліброваною ймовірністю','модель навчалась на всіх доступних пацієнтах']) { const r=await p.inspect({kind:'slide,textbox,notes',search:q,maxChars:1200}); console.log(q, r.ndjson.includes(q)); }

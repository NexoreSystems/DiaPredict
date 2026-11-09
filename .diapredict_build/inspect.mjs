import { FileBlob, PresentationFile } from 'file:///C:/Users/Admin/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs';
const p=await PresentationFile.importPptx(await FileBlob.load('C:/Users/Admin/Downloads/DiaPredict_EHR_Analytics_v9.pptx'));
const s=p.slides.items[10];
for (const [i,x] of s.shapes.items.entries()) console.log(i,x.name,JSON.stringify(x.position),JSON.stringify(x.text?.toString?.()??''),JSON.stringify(x.text?.style??{}));

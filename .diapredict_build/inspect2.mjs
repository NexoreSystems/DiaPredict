import { FileBlob, PresentationFile } from 'file:///C:/Users/Admin/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs';
const p=await PresentationFile.importPptx(await FileBlob.load('C:/Users/Admin/Downloads/DiaPredict_EHR_Analytics_v9.pptx'));
const r=await p.inspect({kind:'slide,textbox,notes',maxChars:100000});
for(const line of r.ndjson.split('\n')){try{const x=JSON.parse(line); if(['slide','notes'].includes(x.kind)|| (x.kind==='textbox'&&[1,2,3,4,5,6,7,8,9,10].includes(x.slideIndex))) console.log(JSON.stringify(x));}catch{}}

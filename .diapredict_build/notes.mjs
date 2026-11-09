import { FileBlob, PresentationFile } from 'file:///C:/Users/Admin/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs';
const p=await PresentationFile.importPptx(await FileBlob.load('C:/Users/Admin/Downloads/DiaPredict_EHR_Analytics_v9.pptx'));
const r=await p.inspect({kind:'notes',maxChars:30000});
for(const line of r.ndjson.split('\n')){try{const x=JSON.parse(line); console.log(JSON.stringify({id:x.id,slideIndex:x.slideIndex,text:x.text}));}catch{}}

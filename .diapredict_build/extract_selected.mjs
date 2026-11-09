import { FileBlob, PresentationFile } from 'file:///C:/Users/Admin/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs';
const p=await PresentationFile.importPptx(await FileBlob.load('C:/Users/Admin/Downloads/DiaPredict_EHR_Analytics_v9.pptx'));
for(const idx of [2,4,5,6,7,8]){
 const s=p.slides.items[idx]; console.log('\n## SLIDE',idx+1);
 const r=await p.inspect({kind:'slide,textbox,notes',search:s.title,maxChars:14000});
 for(const line of r.ndjson.split('\n')){try{const x=JSON.parse(line); if(x.slideIndex===idx) console.log(x.kind==='textbox'?JSON.stringify({id:x.id,text:x.text,position:x.position,style:x.style}):JSON.stringify(x));}catch{}}
}

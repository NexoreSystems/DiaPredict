import { FileBlob, PresentationFile } from 'file:///C:/Users/Admin/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs';
const p=await PresentationFile.importPptx(await FileBlob.load('C:/Users/Admin/Downloads/DiaPredict_EHR_Analytics_v9.pptx'));
for(const idx of [4,5,6,7,8]){
 const s=p.slides.items[idx]; console.log('\n## SLIDE',idx+1);
 for(const x of s.shapes.items){const t=x.text?.toString?.()??''; if(t) console.log(JSON.stringify({text:t,position:x.position, name:x.name}));}
}

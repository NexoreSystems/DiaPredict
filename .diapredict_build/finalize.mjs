import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { FileBlob, PresentationFile } from 'file:///C:/Users/Admin/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs';
const skill='C:/Users/Admin/.codex/plugins/cache/openai-primary-runtime/presentations/26.1105.12137/skills/presentations';
const workspaceDir='C:/Users/Admin/OneDrive/Рабочий стол/Dowload_datasets';
const candidatePath=workspaceDir+'/.diapredict_build/review-candidate.pptx';
const finalPath=workspaceDir+'/deliverables/DiaPredict_EHR_Analytics_v11_reviewed.pptx';
const presentation=await PresentationFile.importPptx(await FileBlob.load(candidatePath));
const { finalizePresentation }=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const result=await finalizePresentation({
 workspaceDir,candidatePath,finalPath,
 pythonExecutable:'C:/Users/Admin/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],
 requiredNativeTableOwnerSlides:[],
 fontPolicy:{basis:'reference',families:['Inter'],referencePath:'C:/Users/Admin/Downloads/DiaPredict_EHR_Analytics_v9.pptx',referenceSha256:'9cdacae98301cebcb7a1330a7b8c251bf948494d8e0fc38c372936cd307debdb'},
 verifyArtifactToolImport:true,
 receiptPath:workspaceDir+'/.diapredict_build/validation-v11.json',
});
console.log(JSON.stringify(result));








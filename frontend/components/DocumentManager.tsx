'use client';
import { FileText, Plus, Trash2, CheckCircle2 } from 'lucide-react';
import type { DocumentInfo } from '@/lib/types';

export function DocumentManager({documents,selected,onToggle,onUpload,onDelete}:{documents:DocumentInfo[];selected:string[];onToggle:(id:string)=>void;onUpload:(f?:File)=>void;onDelete:(id:string)=>void}){
 return <div className="docManager">
   <div className="sectionHeader"><div><b>Documents</b><span>{documents.length} indexed papers</span></div><label className="iconButton"><Plus size={16}/><input hidden type="file" accept="application/pdf" onChange={e=>onUpload(e.target.files?.[0])}/></label></div>
   <div className="docList">{documents.map(d=><button key={d.id} className={`docRow ${selected.includes(d.id)?'selected':''}`} onClick={()=>onToggle(d.id)}>
     <FileText size={17}/><span><b>{d.filename}</b><small>{d.page_count} pages · {d.status}</small></span>{selected.includes(d.id)&&<CheckCircle2 size={16}/>}<i onClick={e=>{e.stopPropagation();onDelete(d.id)}}><Trash2 size={14}/></i>
   </button>)}</div>
 </div>;
}

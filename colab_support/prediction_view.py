"""Export real predictions to CSV, annotated text and a standalone interactive SVG graph."""

import html
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from labels import ENTITY_TAXONOMY, REL_VN_NAMES
from data_pipeline import save_json


def _marked_source(text, entities):
    events = {}
    for e in entities:
        events.setdefault(e["start"], []).append((1,e["id"]))
        events.setdefault(e["end"], []).append((-1,e["id"]))
    by_id={e["id"]:e for e in entities}
    points=sorted({0,len(text),*events})
    active=set(); parts=[]
    for a,b in zip(points,points[1:]):
        for action,eid in events.get(a,[]):
            if action==1: active.add(eid)
            else: active.discard(eid)
        value=html.escape(text[a:b])
        title=html.escape(" | ".join(f"{i}: {by_id[i]['code']}" for i in sorted(active)))
        tag="mark" if active else "span"
        parts.append(f'<{tag} id="offset-{a}" title="{title}">{value}</{tag}>')
    return "".join(parts)


def prediction_dashboard(prediction, title="Kết quả nhận diện", source_info=None):
    entities=prediction["entities"]; relations=prediction["relations"]
    # The full JSON/CSV remains complete even for very large predicted graphs.
    view_relations=sorted(relations,key=lambda r:-r["score"])[:20000]
    payload={"entities":entities,"relations":view_relations,"relation_names":REL_VN_NAMES,
        "entity_names":{k:v[1] for k,v in ENTITY_TAXONOMY.items()},"total_relations":len(relations)}
    data=json.dumps(payload,ensure_ascii=False).replace("<","\\u003c").replace(">","\\u003e").replace("&","\\u0026")
    source_note=(source_info or {}).get("evaluation_note", "Đây là dự đoán chưa được đối chiếu với nhãn chuẩn.")
    template=r'''<!doctype html><html lang="vi"><meta charset="utf-8"><title>__TITLE__</title>
<style>body{font:15px Arial;margin:20px;color:#172033;background:#f5f7fb}h1{font-size:23px}.panel{background:white;padding:16px;border:1px solid #ddd;border-radius:8px;margin:12px 0}button,select,input{padding:7px;margin:4px}#graph{width:100%;height:650px;background:#fff;border:1px solid #ccc;touch-action:none}#source{white-space:pre-wrap;line-height:1.7;max-height:520px;overflow:auto}mark{background:#fff1a6}mark:target{background:#fb923c}.small{color:#475569;font-size:13px}.node{cursor:grab}.edge-label{font-size:11px;paint-order:stroke;stroke:white;stroke-width:4px;stroke-linejoin:round;fill:#334155}.node text{font-size:12px;pointer-events:none}#detail{white-space:pre-wrap;max-height:300px;overflow:auto}a{color:#1d4ed8}</style>
<h1>__TITLE__</h1><div class="panel">__NOTE__<br><span class="small">Điểm mô hình chưa được hiệu chỉnh thành xác suất đúng. Cạnh trong hình là quan hệ dự đoán, không phải cạnh vị trí đầu vào RGCN. Không có nhãn chuẩn thì không tính được Precision/Recall/F1.</span></div>
<div class="panel"><label>Quan hệ <select id="relation"><option value="">Tất cả</option></select></label>
<label>Điểm tối thiểu <input id="score" type="number" min="0" max="1" step=".05" value="0"></label>
<label>Số nút tối đa <input id="nodes" type="number" min="2" max="200" value="45"></label>
<label>Số cạnh tối đa <input id="edges" type="number" min="1" max="500" value="80"></label>
<button id="draw">Áp dụng</button><button id="reset">Đặt lại góc nhìn</button><button id="svgSave">Tải SVG</button>
<div id="counts" class="small"></div><div class="small">Cuộn để phóng to; kéo nền để di chuyển; kéo nút để sắp xếp; bấm nút để xem văn bản và các quan hệ.</div>
<svg id="graph" viewBox="0 0 1200 650"><defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10Z" fill="#64748b"/></marker></defs><g id="scene"></g></svg></div>
<div class="panel"><b>Chi tiết thực thể và quan hệ</b><div id="detail">Chọn một nút trên đồ thị.</div></div>
<div class="panel"><b>Văn bản nguồn và các thực thể được đánh dấu</b><div class="small">Di chuột lên vùng tô màu để xem nhãn. Thực thể chồng lấn có thể cùng xuất hiện trong một vùng.</div><div id="source">__SOURCE__</div></div>
<script type="application/json" id="data">__DATA__</script><script>
const D=JSON.parse(document.getElementById('data').textContent), byId=new Map(D.entities.map(e=>[e.id,e]));
const svg=document.getElementById('graph'),scene=document.getElementById('scene'),NS='http://www.w3.org/2000/svg';
let view=[0,0,1200,650],points=[],links=[],moving=null,dragged=false;
function el(tag,attrs,parent){const n=document.createElementNS(NS,tag);for(const [k,v] of Object.entries(attrs||{}))n.setAttribute(k,v);if(parent)parent.appendChild(n);return n;}
function label(code){return D.relation_names[code]||code;}
function updateView(){svg.setAttribute('viewBox',view.join(' '));}
function details(e){const box=document.getElementById('detail');box.textContent=e.id+' — '+(D.entity_names[e.code]||e.code)+'\nVị trí ['+e.start+', '+e.end+')\nĐiểm NER: '+e.score.toFixed(3)+'\n\n'+e.text+'\n\n';
 const related=D.relations.filter(r=>r.from_id===e.id||r.to_id===e.id);box.appendChild(document.createTextNode('Hiển thị '+Math.min(100,related.length)+'/'+related.length+' quan hệ trong HTML; CSV giữ đầy đủ.\n'));for(const r of related.slice(0,100)){box.appendChild(document.createTextNode(r.from_id+' → '+label(r.code)+' → '+r.to_id+' ('+r.score.toFixed(3)+')\n'));}
 const target=document.getElementById('offset-'+e.start);if(target){document.querySelectorAll('mark').forEach(n=>n.style.outline='');target.style.outline='3px solid #f97316';target.scrollIntoView({block:'center',behavior:'smooth'});}}
function border(p,dx,dy){const k=Math.min(85/Math.max(.001,Math.abs(dx)),27/Math.max(.001,Math.abs(dy)));return {x:p.x+dx*k,y:p.y+dy*k};}
function paths(){for(const l of links){const a=l.a,b=l.b,dx=b.x-a.x,dy=b.y-a.y;let bend=l.parallel?35:0;const cx=(a.x+b.x)/2-dy*.12*bend/35,cy=(a.y+b.y)/2+dx*.12*bend/35;const start=border(a,cx-a.x,cy-a.y),end=border(b,cx-b.x,cy-b.y);l.path.setAttribute('d',`M${start.x},${start.y} Q${cx},${cy} ${end.x},${end.y}`);l.text.setAttribute('x',cx);l.text.setAttribute('y',cy-6);}for(const p of points)p.group.setAttribute('transform',`translate(${p.x},${p.y})`);}
function draw(){scene.replaceChildren();const type=document.getElementById('relation').value,score=Number(document.getElementById('score').value)||0;
 const maxN=Math.max(2,Math.min(200,Number(document.getElementById('nodes').value)||45)),maxE=Math.max(1,Math.min(500,Number(document.getElementById('edges').value)||80));
 const eligible=D.relations.filter(r=>(!type||r.code===type)&&r.score>=score&&byId.has(r.from_id)&&byId.has(r.to_id));const ids=new Set(),chosen=[],groups=new Map();
 for(const r of eligible){const key=r.from_id+'|'+r.to_id;if(!groups.has(key))groups.set(key,{...r,codes:[]});groups.get(key).codes.push(r.code);}
 for(const r of groups.values()){const extra=Number(!ids.has(r.from_id))+Number(!ids.has(r.to_id));if(ids.size+extra>maxN)continue;ids.add(r.from_id);ids.add(r.to_id);chosen.push(r);if(chosen.length>=maxE)break;}
 if(!ids.size)for(const e of D.entities.slice(0,maxN))ids.add(e.id);
 const degree=new Map([...ids].map(id=>[id,0]));for(const r of chosen){degree.set(r.from_id,degree.get(r.from_id)+1);degree.set(r.to_id,degree.get(r.to_id)+1);}
 points=[...ids].map((id,i)=>({id,x:600+240*Math.cos(i*2*Math.PI/ids.size),y:320+210*Math.sin(i*2*Math.PI/ids.size)}));const map=new Map(points.map(p=>[p.id,p]));
 for(let t=0;t<150;t++){const f=new Map(points.map(p=>[p.id,{x:0,y:0}]));for(let i=0;i<points.length;i++)for(let j=i+1;j<points.length;j++){const a=points[i],b=points[j],dx=a.x-b.x,dy=a.y-b.y,d=Math.max(20,Math.hypot(dx,dy)),k=1600/(d*d);f.get(a.id).x+=dx*k;f.get(a.id).y+=dy*k;f.get(b.id).x-=dx*k;f.get(b.id).y-=dy*k;}
 for(const r of chosen){const a=map.get(r.from_id),b=map.get(r.to_id),dx=b.x-a.x,dy=b.y-a.y,d=Math.max(1,Math.hypot(dx,dy)),k=(d-145)*.003;f.get(a.id).x+=dx*k;f.get(a.id).y+=dy*k;f.get(b.id).x-=dx*k;f.get(b.id).y-=dy*k;}
 for(const p of points){p.x=Math.max(100,Math.min(1100,p.x+Math.max(-5,Math.min(5,f.get(p.id).x))));p.y=Math.max(65,Math.min(585,p.y+Math.max(-5,Math.min(5,f.get(p.id).y))));}}
 links=chosen.map(r=>{const path=el('path',{fill:'none',stroke:r.codes.length===1&&r.code==='THUOC'?'#94a3b8':'#2563eb','stroke-width':1.6,'marker-end':'url(#arrow)'},scene),txt=el('text',{'class':'edge-label','text-anchor':'middle'},scene);txt.textContent=label(r.code).slice(0,22)+(r.codes.length>1?' +'+(r.codes.length-1):'');const title=el('title',{},path);title.textContent=r.codes.map(label).join('\n')+' / max score '+r.score.toFixed(3);return {a:map.get(r.from_id),b:map.get(r.to_id),path,text:txt,parallel:chosen.some(s=>s.from_id===r.to_id&&s.to_id===r.from_id)};});
 for(const p of points){const e=byId.get(p.id),g=el('g',{'class':'node'},scene);p.group=g;el('rect',{x:-82,y:-24,width:164,height:48,rx:8,fill:e.code==='DIEU'||e.code==='KHOAN'||e.code==='DIEM'?'#dbeafe':'#dcfce7',stroke:'#475569'},g);const a=el('text',{'text-anchor':'middle',y:-4},g);a.textContent=p.id+' · '+(D.entity_names[e.code]||e.code).slice(0,19);const b=el('text',{'text-anchor':'middle',y:14},g);b.textContent=e.text.replace(/\s+/g,' ').slice(0,23);el('title',{},g).textContent=e.code+'\n'+e.text;
 g.addEventListener('pointerdown',ev=>{ev.stopPropagation();moving={point:p,x:ev.clientX,y:ev.clientY};dragged=false;svg.setPointerCapture(ev.pointerId);});g.addEventListener('click',()=>{if(!dragged)details(e);});}
 paths();document.getElementById('counts').textContent=`Toàn bộ: ${D.entities.length} thực thể, ${D.total_relations} quan hệ. Khớp bộ lọc trong HTML: ${eligible.length}. Đang vẽ ${points.length} nút / ${links.length} cặp có hướng (gộp nhiều nhãn/cặp). JSON/CSV giữ đầy đủ kết quả; HTML xét tối đa 20.000 quan hệ có điểm cao nhất.`;}
for(const code of [...new Set(D.relations.map(r=>r.code))].sort()){const o=document.createElement('option');o.value=code;o.textContent=label(code);document.getElementById('relation').appendChild(o);}
document.getElementById('draw').onclick=draw;document.getElementById('reset').onclick=()=>{view=[0,0,1200,650];updateView();};
svg.addEventListener('wheel',e=>{e.preventDefault();const k=e.deltaY>0?1.12:1/1.12;view=[view[0]+view[2]*(1-k)/2,view[1]+view[3]*(1-k)/2,view[2]*k,view[3]*k];updateView();},{passive:false});
svg.addEventListener('pointerdown',e=>{moving={x:e.clientX,y:e.clientY};svg.setPointerCapture(e.pointerId);});
svg.addEventListener('pointermove',e=>{if(!moving)return;const dx=(e.clientX-moving.x)*view[2]/svg.clientWidth,dy=(e.clientY-moving.y)*view[3]/svg.clientHeight;if(Math.abs(dx)+Math.abs(dy)>1)dragged=true;if(moving.point){moving.point.x+=dx;moving.point.y+=dy;paths();}else{view[0]-=dx;view[1]-=dy;updateView();}moving.x=e.clientX;moving.y=e.clientY;});
svg.addEventListener('pointerup',()=>{if(moving&&moving.point&&!dragged)details(byId.get(moving.point.id));moving=null;});svg.addEventListener('pointercancel',()=>moving=null);
document.getElementById('svgSave').onclick=()=>{const blob=new Blob([new XMLSerializer().serializeToString(svg)],{type:'image/svg+xml'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='relation_graph.svg';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};draw();
</script></html>'''
    return template.replace("__TITLE__",html.escape(title)).replace("__NOTE__",html.escape(source_note)).replace("__SOURCE__",_marked_source(prediction["full_text"],entities)).replace("__DATA__",data)


def export_prediction(prediction, output, source_info=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    save_json(output/"prediction.json",prediction)
    save_json(output/"source_info.json",source_info or {})
    (output/"input_text.txt").write_text(prediction["full_text"],encoding="utf-8")
    pd.DataFrame(prediction["entities"],columns=["id","code","start","end","text","score"]).to_csv(output/"entities.csv",index=False,encoding="utf-8-sig")
    by_id={e["id"]:e for e in prediction["entities"]}
    relations=[{**r,"relation_name":REL_VN_NAMES.get(r["code"],r["code"]),"from_text":by_id[r["from_id"]]["text"],"to_text":by_id[r["to_id"]]["text"]} for r in prediction["relations"]]
    pd.DataFrame(relations,columns=["from_id","to_id","code","relation_name","score","from_text","to_text"]).to_csv(output/"relations.csv",index=False,encoding="utf-8-sig")
    path=output/"prediction.html"
    path.write_text(prediction_dashboard(prediction,source_info=source_info),encoding="utf-8")
    # A compact static preview complements the interactive full report.
    graph=nx.DiGraph()
    for r in sorted(prediction["relations"],key=lambda r:-r["score"]):
        if len(set(graph.nodes)|{r["from_id"],r["to_id"]})>25:continue
        graph.add_edge(r["from_id"],r["to_id"],label=REL_VN_NAMES.get(r["code"],r["code"]))
        if graph.number_of_edges()>=35:break
    if not graph:
        graph.add_nodes_from([e["id"] for e in prediction["entities"][:25]])
    fig,ax=plt.subplots(figsize=(15,10))
    if graph:
        pos=nx.spring_layout(graph,seed=42,k=1.5)
        labels={i:f"{i}: {by_id[i]['code']}\n{by_id[i]['text'][:25]}" for i in graph}
        nx.draw_networkx(graph,pos,labels=labels,node_color="#dbeafe",node_size=1800,font_size=7,arrows=True,ax=ax)
        nx.draw_networkx_edge_labels(graph,pos,edge_labels=nx.get_edge_attributes(graph,"label"),font_size=6,ax=ax)
    else:ax.text(.5,.5,"Không có thực thể/quan hệ dự đoán",ha="center",va="center",transform=ax.transAxes)
    ax.set_title("Quan hệ dự đoán — bản xem gọn; xem HTML/CSV để kiểm tra đầy đủ")
    ax.axis("off");fig.tight_layout();fig.savefig(output/"relation_graph.png",dpi=180);plt.close(fig)
    return path

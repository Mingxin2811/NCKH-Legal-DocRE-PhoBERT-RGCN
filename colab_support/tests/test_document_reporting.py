import json
import sys
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use('Agg')
from document_io import read_input_document
from prediction_view import export_prediction
from data_pipeline import candidate_pairs, split_documents, PIPELINE_VERSION, digest


class DocumentReportingTests(unittest.TestCase):
    def test_docx_preserves_paragraph_table_order_and_utf8(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'law.docx'
            content='''<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>Điều 1</w:t></w:r></w:p><w:tbl><w:tr><w:tc><w:p><w:r><w:t>Cột A</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>Cột B</w:t></w:r></w:p></w:tc></w:tr></w:tbl><w:p><w:r><w:t>Điều 2</w:t></w:r></w:p></w:body></w:document>'''
            with ZipFile(path,'w') as archive:archive.writestr('word/document.xml',content)
            text,info=read_input_document(path)
            self.assertEqual(text,'Điều 1\nCột A\tCột B\nĐiều 2')
            self.assertEqual(info['characters'],len(text))

    def test_export_escapes_html_keeps_all_predictions_and_handles_empty(self):
        prediction={'full_text':'A <script>alert(1)</script> B','entities':[
            {'id':'P0','code':'DIEU','start':0,'end':1,'text':'A','score':.8},
            {'id':'P1','code':'KHOAN','start':28,'end':29,'text':'B','score':.7}],
            'relations':[{'from_id':'P1','to_id':'P0','code':'THUOC','score':.6}]}
        with tempfile.TemporaryDirectory() as tmp:
            path=export_prediction(prediction,tmp)
            text=path.read_text(encoding='utf8')
            self.assertNotIn('<script>alert(1)</script>',text)
            self.assertIn('&lt;script&gt;',text)
            self.assertEqual(json.loads((Path(tmp)/'prediction.json').read_text(encoding='utf8')),prediction)
            for name in ['entities.csv','relations.csv','input_text.txt','relation_graph.png']:self.assertTrue((Path(tmp)/name).is_file())
            export_prediction({'full_text':'Không có kết quả','entities':[],'relations':[]},Path(tmp)/'empty')

    def test_enclosing_candidates_do_not_read_entity_codes(self):
        entities=[{'start':0,'end':100,'code':'DIEU'},{'start':30,'end':40,'code':'KHOAN'}, {'start':101,'end':105,'code':'SO_HIEU'}]
        pairs=candidate_pairs(entities,0,0,1)
        self.assertEqual(pairs,[(0,1),(1,0)])
        self.assertEqual(pairs,candidate_pairs([dict(e,code='NGUOI_KY') for e in entities],0,0,1))

    def test_saved_leaky_manifest_is_rejected(self):
        docs=[{'id':str(i),'text':str(i),'entities':[],'relations':[]} for i in range(3)]
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'split.json'
            split_documents(docs,path)
            manifest=json.loads(path.read_text(encoding='utf8'))
            manifest['splits']['val']=manifest['splits']['train']
            path.write_text(json.dumps(manifest),encoding='utf8')
            with self.assertRaisesRegex(ValueError,'overlap'):split_documents(docs,path)


if __name__=='__main__':unittest.main()

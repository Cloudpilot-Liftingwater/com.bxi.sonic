"""MM ABI through official writer and pause cleanup, offline synthetic input."""
from pathlib import Path
import sys,json,tempfile,unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import official_writer
from contract import VECTOR_NAMES,CAMERAS,CONTRACT_ID
from export_dataset import export
import pyarrow.parquet as pq

class MillimeterDataset(unittest.TestCase):
 def test_mm_roundtrip_pause_and_discard(self):
  with tempfile.TemporaryDirectory() as tmp:
   out=Path(tmp)
   w=official_writer.create(out/'raw','SYNTHETIC OFFLINE TEST ONLY',camera_shapes={k:(32,32,3) for k in CAMERAS},online=True,record_smpl=True)
   try:
    for discarded in (False,True):
     for i in range(120):
      frame={k:np.zeros(len(v),np.float32) for k,v in VECTOR_NAMES.items()}
      frame['observation.state'][-2:]=[20+i*.1,50-i*.1]
      frame['action.wbc'][-2:]=[25+i*.1,55-i*.1]
      frame['action.motion_token'][:]=i/120
      frame['observation.projected_gravity'][:]=[0,0,-1]
      frame['teleop.smpl_pose']=np.full(63,0 if 50<=i<60 else i+1,dtype=np.float32)
      frame.update({'observation.images.'+k:np.full((32,32,3),i,dtype=np.uint8) for k in CAMERAS})
      w.add_frame(frame)
     if discarded:w.save_episode_as_discarded()
     else:w.save_episode()
   finally:official_writer.close(w)
   assert official_writer.verify_videos(w)==720
   result=export([out/'raw'],out/'export',inspection_only=True)
   assert result['training_sequences']==1
   info=json.loads((out/'export/meta/info.json').read_text());assert info['script_config']['contract_id']==CONTRACT_ID
   arr=pq.read_table(next((out/'export/data').rglob('*.parquet'))).to_pandas()
   assert 40<len(arr)<120
   assert all(np.allclose(a[-2:]-b[-2:],5) for a,b in zip(arr['action.wbc'],arr['observation.state']))
   assert np.min(np.stack(arr['observation.state'])[:,-2])>=20
   assert np.max(np.stack(arr['action.wbc'])[:,-1])==55
   print('MM_OFFICIAL_EXPORT_PASS '+json.dumps(result))

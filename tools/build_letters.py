"""Builds ishara/signs-base.json from the letters training images (train-data branch).
Features are computed exactly like IsharaCore.features() in ishara/index.html."""
import json, math, os, sys, base64, io, urllib.request
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions
from PIL import Image

DATA = sys.argv[1]; OUT = sys.argv[2]
MODEL = 'hand_landmarker.task'
urllib.request.urlretrieve('https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task', MODEL)
NAMES = {'Alef':'ا','Beh':'ب','Teh':'ت','Theh':'ث','Jeem':'ج','Hah':'ح','Khah':'خ','Dal':'د','Thal':'ذ','Reh':'ر','Zain':'ز',
 'Seen':'س','Sheen':'ش','Sad':'ص','Dad':'ض','Tah':'ط','Zah':'ظ','Ain':'ع','Ghain':'غ','Feh':'ف','Qaf':'ق','Kaf':'ك','Lam':'ل',
 'Meem':'م','Noon':'ن','Heh':'ه','Waw':'و','Yeh':'ي','Teh_Marbuta':'ة','Al':'ال','Laa':'لام ألف'}

det = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL), running_mode=vision.RunningMode.IMAGE,
    num_hands=1, min_hand_detection_confidence=0.3, min_hand_presence_confidence=0.3))

def d3(a,b): return math.sqrt((a.x-b.x)**2+(a.y-b.y)**2+((a.z or 0)-(b.z or 0))**2)
def feat(lm, handed):
    flip = -1 if handed=='Left' else 1
    w=lm[0]; s=d3(lm[0],lm[9]) or 1e-6; f=[]
    for p in lm: f += [flip*(p.x-w.x)/s, (p.y-w.y)/s, 0.5*((p.z or 0)-(w.z or 0))/s]
    tips=[4,8,12,16,20]
    for i in range(5):
        for j in range(i+1,5): f.append(d3(lm[tips[i]],lm[tips[j]])/s)
    return [round(v,3) for v in f]

manifest = json.load(open(os.path.join(DATA,'manifest.json')))
signs=[]; report={}
for cls, files in manifest.items():
    label = NAMES.get(cls)
    if not label: continue
    samples=[]; thumb=''
    for fn in files:
        path=os.path.join(DATA,'d',cls,fn)
        img = mp.Image.create_from_file(path)
        r = det.detect(img)
        if not r.hand_landmarks: continue
        handed = r.handedness[0][0].category_name
        samples.append(feat(r.hand_landmarks[0], handed))
        if not thumb:
            im=Image.open(path).convert('RGB'); im.thumbnail((180,180)); b=io.BytesIO(); im.save(b,'JPEG',quality=70)
            thumb='data:image/jpeg;base64,'+base64.b64encode(b.getvalue()).decode()
    report[label]=f'{len(samples)}/{len(files)}'
    if samples: signs.append({'label':label,'desc':f'حرف «{label}» بلغة الإشارة العربية','thumb':thumb,'hands':1,'samples':samples})
json.dump({'app':'ishara','v':2,'source':'RGB Arabic Alphabets Sign Language Dataset (Al-Barham et al., 2023), CC BY-NC-SA 4.0','signs':signs},
          open(OUT,'w',encoding='utf-8'), ensure_ascii=False, separators=(',',':'))
print(json.dumps(report, ensure_ascii=False))

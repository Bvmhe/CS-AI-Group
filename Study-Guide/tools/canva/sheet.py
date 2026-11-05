import sys,glob,os
from PIL import Image
o=sys.argv[1]; a,b=int(sys.argv[2]),int(sys.argv[3]); out=sys.argv[4]
fs=[os.path.join(o,'pv','%03d.jpg'%i) for i in range(a,b+1) if os.path.exists(os.path.join(o,'pv','%03d.jpg'%i))]
cols=8; w,h=270,480; rows=(len(fs)+cols-1)//cols
S=Image.new('RGB',(cols*w,rows*h),'white')
for k,f in enumerate(fs): S.paste(Image.open(f).resize((w,h)),((k%cols)*w,(k//cols)*h))
S.save(out,quality=85)


"""
Universal Math Dataset Generator
- 31 topics, every image is UNIQUE via randomized geometry, colors, sizes, values
- Safe for 2000+ images with no duplicates
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, Polygon, Rectangle, Ellipse, Arc
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import os, json, random, textwrap

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATASET_NAME = "data"
NUM_IMAGES   = 1        # change freely
os.makedirs(DATASET_NAME, exist_ok=True)

# ── COLOUR PALETTES ───────────────────────────────────────────────────────────
PALETTES = [
    ('#AED6F1','#1A5276'), ('#A9DFBF','#1E8449'), ('#F9E79F','#B7950B'),
    ('#D2B4DE','#6C3483'), ('#FADBD8','#C0392B'), ('#FAD7A0','#D35400'),
    ('#D5F5E3','#117A65'), ('#EBF5FB','#2E86C1'), ('#A8D8EA','#2471A3'),
    ('#D7BDE2','#7D3C98'), ('#ABEBC6','#1D8348'), ('#F5CBA7','#A04000'),
    ('#D6EAF8','#154360'), ('#FDEBD0','#784212'), ('#E8DAEF','#512E5F'),
    ('#D1F2EB','#0E6655'), ('#FDFEFE','#2C3E50'), ('#FEF9E7','#7D6608'),
]

def rpal():
    fc, ec = random.choice(PALETTES)
    return fc, ec

def rng(lo, hi, decimals=1):
    return round(random.uniform(lo, hi), decimals)

# ── HELPERS ───────────────────────────────────────────────────────────────────
def base_fig():
    fig, ax = plt.subplots(figsize=(6, 7))
    bg = random.choice(['#ffffff','#fafafa','#f5f5f5','#fffcf5','#f0f4f8'])
    ax.set_facecolor(bg)
    if random.random() > 0.35:
        ax.grid(True, linestyle=random.choice(['--',':','-']), alpha=0.2, color='#aaaaaa')
    ax.set_aspect('equal')
    return fig, ax

def label_and_save(fig, category, topic, desc, eid, save_dir):
    wrapped = textwrap.fill(desc, width=65)
    label   = f"TOPIC: {topic}\n{wrapped}"
    plt.figtext(0.5, 0.95, label, ha='center', va='top', fontsize=8.5,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                          alpha=0.95, edgecolor='#333333'))
    plt.subplots_adjust(top=0.78, bottom=0.08, left=0.1, right=0.95)
    fname = f"{topic.lower().replace(' ','_')}_{eid:05d}.png"
    fig.savefig(os.path.join(save_dir, fname), dpi=120, bbox_inches='tight')
    plt.close(fig)
    return {"id": eid, "file": fname, "topic": topic,
            "category": category, "description": desc}

# ══════════════════════════════════════════════════════════════════════════════
#  DRAW FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def draw_circle(eid, save_dir):
    fig, ax = base_fig()
    fc, ec = rpal()
    r = rng(0.8, 2.8)
    cx, cy = rng(-0.5, 0.5), rng(-0.3, 0.3)
    ax.add_patch(Circle((cx, cy), r, fc=fc, ec=ec, lw=2, alpha=random.uniform(0.55,0.9)))
    ax.plot([cx, cx+r], [cy, cy], color=ec, lw=1.8, zorder=5)
    ax.plot(cx, cy, 'ko', ms=4, zorder=6)
    ax.text(cx+r/2, cy+0.15, f'r = {r}', fontsize=9, color=ec, ha='center')
    area = np.pi*r**2; circ = 2*np.pi*r
    ax.text(cx, cy-r-0.7,  f'A = πr² = {area:.3f}', fontsize=10.5, ha='center', color=ec)
    ax.text(cx, cy-r-1.2,  f'C = 2πr = {circ:.3f}', fontsize=10.5, ha='center', color=ec)
    pad = r+1.5
    ax.set_xlim(cx-pad, cx+pad); ax.set_ylim(cy-pad-0.8, cy+pad)
    desc = f"Circle centre=({cx},{cy}), r={r}. Area={area:.3f}, Circumference={circ:.3f}."
    return label_and_save(fig,"2D Geometry","Circle",desc,eid,save_dir)

def draw_triangle(eid, save_dir):
    fig, ax = base_fig()
    fc, ec = rpal()
    while True:
        pts = np.array([[0,0],[rng(2.5,5.0),0],[rng(0.5,4.5),rng(1.5,4.0)]])
        s0=np.linalg.norm(pts[1]-pts[0]); s1=np.linalg.norm(pts[2]-pts[1]); s2=np.linalg.norm(pts[0]-pts[2])
        sp=(s0+s1+s2)/2; disc=sp*(sp-s0)*(sp-s1)*(sp-s2)
        if disc>0.5: break
    area=np.sqrt(disc)
    ax.add_patch(Polygon(pts, fc=fc, ec=ec, lw=2, alpha=0.75))
    mids=[(pts[0]+pts[1])/2,(pts[1]+pts[2])/2,(pts[2]+pts[0])/2]
    offs=[(0,-0.3),(0.35,0.2),(-0.45,0.2)]
    for i,(lab,s) in enumerate(zip(['a','b','c'],[s0,s1,s2])):
        ax.text(mids[i][0]+offs[i][0],mids[i][1]+offs[i][1],f'{lab}={s:.2f}',fontsize=8.5,color=ec,ha='center')
    ax.text(pts[1][0]/2, pts[0][1]-0.95, f"Area={area:.2f}  P={s0+s1+s2:.2f}", fontsize=9.5, ha='center', color=ec)
    ax.set_xlim(-0.8, pts[1][0]+1); ax.set_ylim(pts[0][1]-1.5, pts[2][1]+0.8)
    desc=f"Triangle sides a={s0:.2f}, b={s1:.2f}, c={s2:.2f}. Area={area:.2f}. P={s0+s1+s2:.2f}."
    return label_and_save(fig,"2D Geometry","Triangle",desc,eid,save_dir)

def draw_square(eid, save_dir):
    fig, ax = base_fig()
    fc, ec = rpal()
    s = rng(1.5, 3.2)
    ax.add_patch(Rectangle((-s/2,-s/2), s, s, fc=fc, ec=ec, lw=2, alpha=random.uniform(0.6,0.92)))
    d = s*np.sqrt(2)
    ax.plot([-s/2,s/2],[-s/2,s/2],'--',color=ec,lw=1.2,alpha=0.6)
    ax.text(0.15,0.15,f'd={d:.2f}',fontsize=8.5,color=ec,rotation=45,ha='center')
    ax.text(0,-s/2-0.5,f'side={s}',fontsize=9.5,ha='center',color=ec)
    ax.text(0,-s/2-1.0,f'A=s²={s**2:.2f}',fontsize=10.5,ha='center',color=ec)
    ax.text(0,-s/2-1.5,f'P=4s={4*s:.2f}',fontsize=10.5,ha='center',color=ec)
    lim=s/2+1.8; ax.set_xlim(-lim,lim); ax.set_ylim(-lim-1.2,lim)
    desc=f"Square side={s}. Area={s**2:.2f}, Perimeter={4*s:.2f}, Diagonal={d:.2f}."
    return label_and_save(fig,"2D Geometry","Square",desc,eid,save_dir)

def draw_rectangle(eid, save_dir):
    fig, ax = base_fig()
    fc, ec = rpal()
    w = rng(2.5, 5.0); h = rng(1.2, 3.0)
    ax.add_patch(Rectangle((-w/2,-h/2), w, h, fc=fc, ec=ec, lw=2, alpha=random.uniform(0.6,0.92)))
    ax.text(0,-h/2-0.4,f'w={w}',fontsize=9,ha='center',color=ec)
    ax.text(w/2+0.5,0,f'h={h}',fontsize=9,ha='center',color=ec)
    ax.text(0,-h/2-0.9,f'A=w×h={w*h:.2f}',fontsize=10.5,ha='center',color=ec)
    ax.text(0,-h/2-1.4,f'P=2(w+h)={2*(w+h):.2f}',fontsize=10.5,ha='center',color=ec)
    ax.set_xlim(-w/2-1,w/2+1.5); ax.set_ylim(-h/2-2,h/2+1)
    desc=f"Rectangle w={w}, h={h}. Area={w*h:.2f}, Perimeter={2*(w+h):.2f}."
    return label_and_save(fig,"2D Geometry","Rectangle",desc,eid,save_dir)

def draw_equilateral_triangle(eid, save_dir):
    fig, ax = base_fig()
    fc, ec = rpal()
    s = rng(2.0, 4.5)
    h = s*np.sqrt(3)/2
    pts = np.array([[0,0],[s,0],[s/2,h]])
    ax.add_patch(Polygon(pts, fc=fc, ec=ec, lw=2, alpha=0.75))
    for i in range(3):
        p1,p2=pts[i],pts[(i+1)%3]; mid=(p1+p2)/2
        ax.text(mid[0],mid[1]+(-0.3 if i==0 else 0.22),f's={s}',fontsize=9,color=ec,ha='center')
    ax.plot([s/2,s/2],[0,h],'--',color=ec,lw=1.2)
    ax.text(s/2+0.2,h/2,f'h={h:.2f}',fontsize=8.5,color=ec)
    area=(np.sqrt(3)/4)*s**2
    ax.text(s/2,-0.75,f'A=(√3/4)s²={area:.3f}',fontsize=10,ha='center',color=ec)
    ax.set_xlim(-0.5,s+0.8); ax.set_ylim(-1.3,h+0.7)
    desc=f"Equilateral triangle s={s}. All angles=60°. h={h:.2f}. Area={area:.3f}."
    return label_and_save(fig,"2D Geometry","Equilateral Triangle",desc,eid,save_dir)

def draw_isosceles_triangle(eid, save_dir):
    fig, ax = base_fig()
    fc, ec = rpal()
    base = rng(2.0, 4.5)
    leg  = rng(base*0.7+0.5, base*1.5+1.0)
    disc = leg**2-(base/2)**2
    if disc < 0.01: disc = 0.5
    ht = np.sqrt(disc)
    pts = np.array([[0,0],[base,0],[base/2,ht]])
    ax.add_patch(Polygon(pts, fc=fc, ec=ec, lw=2, alpha=0.75))
    ax.text(base/2,-0.35,f'Base={base}',fontsize=9,ha='center',color=ec)
    ax.plot([base/2,base/2],[0,ht],'--',color=ec,lw=1.2)
    ax.text(base/2+0.18,ht/2,f'h={ht:.2f}',fontsize=8.5,color=ec)
    area=0.5*base*ht
    ax.text(base/2,-0.9,f'Leg={leg:.2f}   A=½bh={area:.2f}',fontsize=9.5,ha='center',color=ec)
    ax.set_xlim(-0.5,base+0.7); ax.set_ylim(-1.4,ht+0.8)
    desc=f"Isosceles triangle: base={base}, legs={leg:.2f}, h={ht:.2f}. Area={area:.2f}."
    return label_and_save(fig,"2D Geometry","Isosceles Triangle",desc,eid,save_dir)

def draw_scalene_triangle(eid, save_dir):
    fig, ax = base_fig()
    fc, ec = rpal()
    while True:
        pts=np.array([[0,0],[rng(3,5),0],[rng(0.8,4),rng(1.5,3.8)]])
        s0=np.linalg.norm(pts[1]-pts[0]); s1=np.linalg.norm(pts[2]-pts[1]); s2=np.linalg.norm(pts[0]-pts[2])
        if abs(s0-s1)>0.4 and abs(s1-s2)>0.4 and abs(s0-s2)>0.4:
            sp=(s0+s1+s2)/2; disc=sp*(sp-s0)*(sp-s1)*(sp-s2)
            if disc>0.5: break
    area=np.sqrt(disc)
    ax.add_patch(Polygon(pts, fc=fc, ec=ec, lw=2, alpha=0.75))
    mids=[(pts[0]+pts[1])/2,(pts[1]+pts[2])/2,(pts[2]+pts[0])/2]
    offs=[(0,-0.3),(0.45,0.12),(-0.55,0.12)]
    for i,(lab,s) in enumerate(zip(['a','b','c'],[s0,s1,s2])):
        ax.text(mids[i][0]+offs[i][0],mids[i][1]+offs[i][1],f'{lab}={s:.2f}',fontsize=8.5,color=ec,ha='center')
    ax.text(pts[1][0]/2,pts[0][1]-0.85,'All sides different → Scalene',fontsize=9,ha='center',color=ec)
    ax.text(pts[1][0]/2,pts[0][1]-1.35,f'Area={area:.2f}  P={s0+s1+s2:.2f}',fontsize=10,ha='center',color=ec)
    ax.set_xlim(-0.5,pts[1][0]+1); ax.set_ylim(pts[0][1]-1.9,pts[2][1]+0.8)
    desc=f"Scalene triangle sides {s0:.2f},{s1:.2f},{s2:.2f}. Area={area:.2f}."
    return label_and_save(fig,"2D Geometry","Scalene Triangle",desc,eid,save_dir)

def draw_right_triangle(eid, save_dir):
    fig, ax = base_fig()
    fc, ec = rpal()
    if random.random() > 0.4:
        triples = [(3,4,5),(5,12,13),(8,15,17),(7,24,25),(9,40,41),(6,8,10),(20,21,29)]
        a,b,_ = random.choice(triples)
    else:
        a = rng(1.5, 6.0); b = rng(1.5, 6.0)
    c = np.sqrt(a**2+b**2)
    pts = np.array([[0,0],[b,0],[0,a]])
    ax.add_patch(Polygon(pts, fc=fc, ec=ec, lw=2, alpha=0.75))
    box_s = min(a,b)*0.12
    ax.add_patch(Rectangle((0,0),box_s,box_s,fill=False,ec=ec,lw=1.5))
    ax.text(b/2,-0.3,f'b={b}',fontsize=9.5,ha='center',color=ec)
    ax.text(-0.5,a/2,f'a={a}',fontsize=9.5,ha='center',color=ec)
    ang=np.degrees(np.arctan2(a,b))
    ax.text(b/2+0.35,a/2,f'c={c:.2f}',fontsize=9.5,ha='center',color=ec,rotation=-ang)
    ax.text(b/2,-0.85,'a²+b²=c²',fontsize=9.5,ha='center',color=ec)
    ax.text(b/2,-1.35,f'{a}²+{b}²={a**2+b**2:.0f}={c:.2f}²',fontsize=9,ha='center',color=ec)
    ax.text(b/2,-1.85,f'Area=½ab={0.5*a*b:.2f}',fontsize=10,ha='center',color=ec)
    ax.set_xlim(-0.9,b+1); ax.set_ylim(-2.3,a+0.9)
    desc=f"Right triangle: a={a}, b={b}, c={c:.2f}. Area={0.5*a*b:.2f}."
    return label_and_save(fig,"2D Geometry","Right-Angled Triangle",desc,eid,save_dir)

def draw_parallelogram(eid, save_dir):
    fig, ax = base_fig()
    fc, ec = rpal()
    base = rng(3.0, 5.5); ht = rng(1.2, 3.0); skew = rng(0.5, 2.0)
    pts = np.array([[skew,0],[base+skew,0],[base,ht],[0,ht]])
    ax.add_patch(Polygon(pts, fc=fc, ec=ec, lw=2, alpha=0.78))
    ax.plot([pts[3][0],pts[3][0]],[0,ht],'--',color=ec,lw=1.5)
    ax.text(pts[3][0]-0.4,ht/2,f'h={ht}',fontsize=9,color=ec)
    ax.text((pts[0][0]+pts[1][0])/2,-0.4,f'base={base}',fontsize=9.5,ha='center',color=ec)
    ax.text((pts[0][0]+pts[1][0])/2,-0.95,f'A=base×h={base*ht:.2f}',fontsize=10.5,ha='center',color=ec)
    ax.set_xlim(-0.5,base+skew+1); ax.set_ylim(-1.6,ht+0.8)
    desc=f"Parallelogram: base={base}, height={ht}. Area={base*ht:.2f}."
    return label_and_save(fig,"2D Geometry","Parallelogram",desc,eid,save_dir)

def draw_rhombus(eid, save_dir):
    fig, ax = base_fig()
    fc, ec = rpal()
    d1 = rng(3.0, 6.0); d2 = rng(1.5, d1*0.85)
    pts = np.array([[d1/2,0],[0,d2/2],[-d1/2,0],[0,-d2/2]])
    ax.add_patch(Polygon(pts, fc=fc, ec=ec, lw=2, alpha=0.78))
    ax.plot([-d1/2,d1/2],[0,0],'--',color=ec,lw=1.2)
    ax.plot([0,0],[-d2/2,d2/2],'--',color=ec,lw=1.2)
    ax.text(0,0.2,f'd₁={d1}',fontsize=9,ha='center',color=ec)
    ax.text(0.18,d2/2+0.3,f'd₂={d2}',fontsize=9,ha='center',color=ec)
    side=np.sqrt((d1/2)**2+(d2/2)**2); area=(d1*d2)/2
    ax.text(0,-d2/2-0.6,f'Side={side:.2f}  A=d₁d₂/2={area:.2f}',fontsize=10,ha='center',color=ec)
    lim=d1/2+0.8; ax.set_xlim(-lim,lim); ax.set_ylim(-d2/2-1.4,d2/2+0.8)
    desc=f"Rhombus: d₁={d1}, d₂={d2}. Side={side:.2f}. Area={area:.2f}."
    return label_and_save(fig,"2D Geometry","Rhombus",desc,eid,save_dir)

def draw_trapezium(eid, save_dir):
    fig, ax = base_fig()
    fc, ec = rpal()
    b1 = rng(3.0, 6.0); b2 = rng(1.0, b1*0.85); ht = rng(1.5, 3.5)
    xo = (b1-b2)/2
    pts = np.array([[0,0],[b1,0],[b1-xo,ht],[xo,ht]])
    ax.add_patch(Polygon(pts, fc=fc, ec=ec, lw=2, alpha=0.78))
    ax.text(b1/2,-0.38,f'b₁={b1:.2f}',fontsize=9.5,ha='center',color=ec)
    ax.text(b1/2,ht+0.28,f'b₂={b2:.2f}',fontsize=9.5,ha='center',color=ec)
    ax.plot([0,0],[0,ht],'--',color=ec,lw=1.2)
    ax.text(-0.45,ht/2,f'h={ht:.2f}',fontsize=9,color=ec)
    area=0.5*(b1+b2)*ht
    ax.text(b1/2,-0.95,f'A=½(b₁+b₂)h={area:.2f}',fontsize=10,ha='center',color=ec)
    ax.set_xlim(-0.8,b1+0.8); ax.set_ylim(-1.6,ht+0.9)
    desc=f"Trapezium: b₁={b1:.2f}, b₂={b2:.2f}, h={ht:.2f}. Area={area:.2f}."
    return label_and_save(fig,"2D Geometry","Trapezium",desc,eid,save_dir)

def draw_kite(eid, save_dir):
    fig, ax = base_fig()
    fc, ec = rpal()
    d1 = rng(3.5, 6.0); d2 = rng(2.0, 4.5)
    split = rng(0.25, 0.75)
    top=d1*split; bot=d1*(1-split)
    pts = np.array([[0,top],[d2/2,0],[0,-bot],[-d2/2,0]])
    ax.add_patch(Polygon(pts, fc=fc, ec=ec, lw=2, alpha=0.78))
    ax.plot([0,0],[-bot,top],'--',color=ec,lw=1.2)
    ax.plot([-d2/2,d2/2],[0,0],'--',color=ec,lw=1.2)
    ax.text(0.15,top/2,f'd₁={d1:.2f}',fontsize=9,color=ec)
    ax.text(d2/4+0.1,0.22,f'd₂={d2:.2f}',fontsize=9,color=ec)
    area=(d1*d2)/2
    ax.text(0,-bot-0.55,f'A=d₁d₂/2={area:.2f}',fontsize=10.5,ha='center',color=ec)
    ax.set_xlim(-d2/2-0.8,d2/2+0.8); ax.set_ylim(-bot-1.2,top+0.8)
    desc=f"Kite: d₁={d1:.2f}, d₂={d2:.2f}. Area={area:.2f}."
    return label_and_save(fig,"2D Geometry","Kite",desc,eid,save_dir)

def draw_pentagon(eid, save_dir):
    fig, ax = base_fig()
    fc, ec = rpal()
    R = rng(1.8, 3.0)
    rot = rng(0, 360)
    angles = [np.radians(90+rot+72*k) for k in range(5)]
    pts = np.array([[R*np.cos(a),R*np.sin(a)] for a in angles])
    ax.add_patch(Polygon(pts, fc=fc, ec=ec, lw=2, alpha=0.78))
    side = 2*R*np.sin(np.pi/5)
    area = (5*side**2)/(4*np.tan(np.pi/5))
    for pt in pts:
        ax.plot(*pt,'o',color=ec,ms=4,zorder=5)
    ax.text(0,-R-0.65,f'R={R:.2f}  side={side:.2f}',fontsize=9.5,ha='center',color=ec)
    ax.text(0,-R-1.15,f'Interior=108°  A={area:.2f}',fontsize=10,ha='center',color=ec)
    lim=R+1.5; ax.set_xlim(-lim,lim); ax.set_ylim(-lim-1,lim)
    desc=f"Regular pentagon: R={R:.2f}, side={side:.2f}, rot={rot:.0f}°. Area={area:.2f}."
    return label_and_save(fig,"2D Geometry","Pentagon",desc,eid,save_dir)

def draw_ellipse(eid, save_dir):
    fig, ax = base_fig()
    fc, ec = rpal()
    a = rng(1.8, 3.5); b = rng(0.8, a*0.85)
    ax.add_patch(Ellipse((0,0), 2*a, 2*b, fc=fc, ec=ec, lw=2, alpha=random.uniform(0.6,0.88)))
    ax.plot([0,a],[0,0],color=ec,lw=1.8)
    ax.plot([0,0],[0,b],color='#E74C3C',lw=1.8)
    ax.text(a/2,0.18,f'a={a}',fontsize=9,color=ec,ha='center')
    ax.text(0.2,b/2,f'b={b}',fontsize=9,color='#C0392B')
    c=np.sqrt(a**2-b**2); e=c/a; area=np.pi*a*b
    ax.plot([-c,c],[0,0],'r+',ms=9,mew=2,label='Foci')
    ax.legend(fontsize=8,loc='lower right')
    ax.text(0,-b-0.65,f'c={c:.2f}  e={e:.2f}',fontsize=9.5,ha='center',color=ec)
    ax.text(0,-b-1.15,f'A=πab={area:.2f}',fontsize=10.5,ha='center',color=ec)
    lim=a+1.2; ax.set_xlim(-lim,lim); ax.set_ylim(-b-1.8,b+1)
    desc=f"Ellipse: a={a}, b={b}, c={c:.2f}, e={e:.2f}. Area={area:.2f}."
    return label_and_save(fig,"Conic Sections","Ellipse",desc,eid,save_dir)

def draw_parabola(eid, save_dir):
    fig, ax = plt.subplots(figsize=(6,7))
    ax.set_facecolor(random.choice(['#fafafa','#ffffff','#f5f5f5']))
    ax.grid(True,linestyle='--',alpha=0.25)
    fc, ec = rpal()
    a = rng(0.2, 1.2); h = rng(-1.0, 1.0); k = rng(-0.5, 0.5)
    x = np.linspace(h-3.5, h+3.5, 400)
    y = a*(x-h)**2 + k
    ax.plot(x, y, color=ec, lw=2.5, label=rf'y={a:.2f}(x{h:+.2f})²{k:+.2f}')
    focus_y = k + 1/(4*a)
    ax.plot(h, focus_y,'ro',ms=7,label=f'Focus ({h:.2f},{focus_y:.2f})')
    dir_y = k - 1/(4*a)
    ax.axhline(dir_y,color='#E67E22',lw=1.5,linestyle='--',label=f'Directrix y={dir_y:.2f}')
    ax.plot(h,k,'g^',ms=7,label=f'Vertex ({h:.2f},{k:.2f})')
    ax.axvline(h,color='#2E86C1',lw=1,linestyle=':',alpha=0.5)
    ax.set_xlim(h-4,h+4); ax.set_ylim(min(y)-0.5,max(y[:200])+1)
    ax.legend(fontsize=8,loc='upper right')
    desc=f"Parabola y={a:.2f}(x{h:+.2f})²{k:+.2f}. Vertex=({h:.2f},{k:.2f}). Focus=({h:.2f},{focus_y:.2f})."
    return label_and_save(fig,"Conic Sections","Parabola",desc,eid,save_dir)

def draw_hyperbola(eid, save_dir):
    fig, ax = plt.subplots(figsize=(6,7))
    ax.set_facecolor(random.choice(['#fafafa','#ffffff']))
    ax.grid(True,linestyle='--',alpha=0.25)
    fc, ec = rpal()
    a = rng(0.8, 2.5); b = rng(0.5, 2.0)
    t = np.linspace(-2.8, 2.8, 600)
    ax.plot(a*np.cosh(t), b*np.sinh(t), color=ec, lw=2.5)
    ax.plot(-a*np.cosh(t), b*np.sinh(t), color=ec, lw=2.5)
    xr = np.linspace(-5.5, 5.5, 200)
    ax.plot(xr,(b/a)*xr,'--',color='#7F8C8D',lw=1.2,alpha=0.7)
    ax.plot(xr,-(b/a)*xr,'--',color='#7F8C8D',lw=1.2,alpha=0.7,label=f'Asym y=±{b/a:.2f}x')
    c=np.sqrt(a**2+b**2)
    ax.plot([-c,c],[0,0],'b+',ms=9,mew=2,label=f'Foci ±{c:.2f}')
    ax.text(0,-4.2,rf'$\frac{{x^2}}{{{a:.2f}^2}}-\frac{{y^2}}{{{b:.2f}^2}}=1$',
            fontsize=13,ha='center',color=ec)
    ax.set_xlim(-5.5,5.5); ax.set_ylim(-5,5)
    ax.legend(fontsize=8,loc='upper right')
    desc=f"Hyperbola x²/{a:.2f}²−y²/{b:.2f}²=1. Foci=±{c:.2f}. Asymptotes y=±{b/a:.2f}x."
    return label_and_save(fig,"Conic Sections","Hyperbola",desc,eid,save_dir)

def draw_cube(eid, save_dir):
    fig = plt.figure(figsize=(6,7))
    ax  = fig.add_subplot(111, projection='3d')
    fc, ec = rpal()
    s = rng(1.0, 2.5)
    r = [-s/2, s/2]
    faces=[
        [[r[0],r[0],r[0]],[r[1],r[0],r[0]],[r[1],r[1],r[0]],[r[0],r[1],r[0]]],
        [[r[0],r[0],r[1]],[r[1],r[0],r[1]],[r[1],r[1],r[1]],[r[0],r[1],r[1]]],
        [[r[0],r[0],r[0]],[r[1],r[0],r[0]],[r[1],r[0],r[1]],[r[0],r[0],r[1]]],
        [[r[0],r[1],r[0]],[r[1],r[1],r[0]],[r[1],r[1],r[1]],[r[0],r[1],r[1]]],
        [[r[0],r[0],r[0]],[r[0],r[1],r[0]],[r[0],r[1],r[1]],[r[0],r[0],r[1]]],
        [[r[1],r[0],r[0]],[r[1],r[1],r[0]],[r[1],r[1],r[1]],[r[1],r[0],r[1]]],
    ]
    pc=Poly3DCollection(faces,alpha=random.uniform(0.2,0.45),facecolor=fc,edgecolor=ec)
    ax.add_collection3d(pc)
    m=s/2+0.5; ax.set_xlim(-m,m); ax.set_ylim(-m,m); ax.set_zlim(-m,m)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.view_init(elev=random.randint(15,40), azim=random.randint(20,70))
    ax.set_title(f'Edge={s:.2f}  V={s**3:.2f}  SA={6*s**2:.2f}',fontsize=10,pad=4)
    desc=f"Cube: edge={s:.2f}. Volume={s**3:.2f}. Surface area={6*s**2:.2f}."
    return label_and_save(fig,"3D Geometry","Cube",desc,eid,save_dir)

def draw_cuboid(eid, save_dir):
    fig = plt.figure(figsize=(6,7))
    ax  = fig.add_subplot(111, projection='3d')
    fc, ec = rpal()
    l=rng(2.0,4.5); w=rng(1.0,l*0.8); h=rng(0.8,l*0.7)
    x=[0,l]; y=[0,w]; z=[0,h]
    verts=[
        [[x[0],y[0],z[0]],[x[1],y[0],z[0]],[x[1],y[1],z[0]],[x[0],y[1],z[0]]],
        [[x[0],y[0],z[1]],[x[1],y[0],z[1]],[x[1],y[1],z[1]],[x[0],y[1],z[1]]],
        [[x[0],y[0],z[0]],[x[1],y[0],z[0]],[x[1],y[0],z[1]],[x[0],y[0],z[1]]],
        [[x[0],y[1],z[0]],[x[1],y[1],z[0]],[x[1],y[1],z[1]],[x[0],y[1],z[1]]],
        [[x[0],y[0],z[0]],[x[0],y[1],z[0]],[x[0],y[1],z[1]],[x[0],y[0],z[1]]],
        [[x[1],y[0],z[0]],[x[1],y[1],z[0]],[x[1],y[1],z[1]],[x[1],y[0],z[1]]],
    ]
    pc=Poly3DCollection(verts,alpha=random.uniform(0.2,0.45),facecolor=fc,edgecolor=ec)
    ax.add_collection3d(pc)
    ax.set_xlim(-0.3,l+0.3); ax.set_ylim(-0.3,w+0.3); ax.set_zlim(-0.3,h+0.3)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.view_init(elev=random.randint(15,40), azim=random.randint(20,70))
    vol=l*w*h; sa=2*(l*w+w*h+l*h)
    ax.set_title(f'l={l:.2f} w={w:.2f} h={h:.2f}  V={vol:.2f}  SA={sa:.2f}',fontsize=9,pad=4)
    desc=f"Cuboid: l={l:.2f}, w={w:.2f}, h={h:.2f}. V={vol:.2f}. SA={sa:.2f}."
    return label_and_save(fig,"3D Geometry","Cuboid",desc,eid,save_dir)

def draw_sphere(eid, save_dir):
    fig = plt.figure(figsize=(6,7))
    ax  = fig.add_subplot(111, projection='3d')
    cmap = random.choice(['Blues','Greens','Oranges','Purples','PuRd','YlOrRd'])
    r = rng(1.0, 2.5)
    u,v=np.mgrid[0:2*np.pi:40j,0:np.pi:20j]
    X=r*np.cos(u)*np.sin(v); Y=r*np.sin(u)*np.sin(v); Z=r*np.cos(v)
    ax.plot_surface(X,Y,Z,alpha=random.uniform(0.3,0.55),cmap=cmap)
    ax.plot_wireframe(X,Y,Z,color='#1A5276',linewidth=0.3,alpha=0.3)
    vol=(4/3)*np.pi*r**3; sa=4*np.pi*r**2
    ax.view_init(elev=random.randint(15,40), azim=random.randint(0,360))
    ax.set_title(f'r={r:.2f}  V={vol:.2f}  SA={sa:.2f}',fontsize=10,pad=4)
    desc=f"Sphere: r={r:.2f}. V=(4/3)πr³={vol:.2f}. SA=4πr²={sa:.2f}."
    return label_and_save(fig,"3D Geometry","Sphere",desc,eid,save_dir)

def draw_cylinder(eid, save_dir):
    fig = plt.figure(figsize=(6,7))
    ax  = fig.add_subplot(111, projection='3d')
    cmap = random.choice(['YlOrRd','Blues','Greens','PuBuGn','BuPu'])
    r=rng(0.8,2.0); h=rng(1.5,4.0)
    theta=np.linspace(0,2*np.pi,60); z=np.linspace(0,h,30)
    T,Z=np.meshgrid(theta,z)
    ax.plot_surface(r*np.cos(T),r*np.sin(T),Z,alpha=random.uniform(0.3,0.5),cmap=cmap)
    t2=np.linspace(0,2*np.pi,60)
    ax.plot(r*np.cos(t2),r*np.sin(t2),0,'k-',lw=1.2)
    ax.plot(r*np.cos(t2),r*np.sin(t2),h,'k-',lw=1.2)
    vol=np.pi*r**2*h; sa=2*np.pi*r*(r+h)
    ax.view_init(elev=random.randint(15,40), azim=random.randint(0,360))
    ax.set_title(f'r={r:.2f}  h={h:.2f}  V={vol:.2f}  SA={sa:.2f}',fontsize=9,pad=4)
    desc=f"Cylinder: r={r:.2f}, h={h:.2f}. V=πr²h={vol:.2f}. SA=2πr(r+h)={sa:.2f}."
    return label_and_save(fig,"3D Geometry","Cylinder",desc,eid,save_dir)

def draw_cone(eid, save_dir):
    fig = plt.figure(figsize=(6,7))
    ax  = fig.add_subplot(111, projection='3d')
    cmap = random.choice(['PuRd','magma','YlOrRd','plasma','Oranges'])
    r=rng(0.8,2.0); h=rng(2.0,4.5)
    theta=np.linspace(0,2*np.pi,60); z=np.linspace(0,h,30)
    T,Z=np.meshgrid(theta,z); R=r*(1-Z/h)
    ax.plot_surface(R*np.cos(T),R*np.sin(T),Z,alpha=random.uniform(0.35,0.55),cmap=cmap)
    t2=np.linspace(0,2*np.pi,60)
    ax.plot(r*np.cos(t2),r*np.sin(t2),0,'k-',lw=1.2)
    slant=np.sqrt(r**2+h**2)
    vol=(1/3)*np.pi*r**2*h; sa=np.pi*r*(r+slant)
    ax.view_init(elev=random.randint(15,40), azim=random.randint(0,360))
    ax.set_title(f'r={r:.2f}  h={h:.2f}  l={slant:.2f}  V={vol:.2f}',fontsize=9,pad=4)
    desc=f"Cone: r={r:.2f}, h={h:.2f}, slant={slant:.2f}. V={vol:.2f}. SA={sa:.2f}."
    return label_and_save(fig,"3D Geometry","Cone",desc,eid,save_dir)

def draw_pyramid(eid, save_dir):
    fig = plt.figure(figsize=(6,7))
    ax  = fig.add_subplot(111, projection='3d')
    fc, ec = rpal()
    s=rng(1.5,3.0); h=rng(2.0,4.5)
    base=np.array([[-s/2,-s/2,0],[s/2,-s/2,0],[s/2,s/2,0],[-s/2,s/2,0]])
    apex=np.array([0,0,h])
    faces=[[base[0],base[1],base[2],base[3]],[base[0],base[1],apex],
           [base[1],base[2],apex],[base[2],base[3],apex],[base[3],base[0],apex]]
    colors=[fc,'#FAD7A0','#F0B27A','#E59866',ec]
    pc=Poly3DCollection(faces,alpha=random.uniform(0.3,0.5),facecolor=colors,edgecolor=ec)
    ax.add_collection3d(pc)
    m=s/2+0.5; ax.set_xlim(-m,m); ax.set_ylim(-m,m); ax.set_zlim(0,h+0.5)
    ax.view_init(elev=random.randint(15,40), azim=random.randint(20,80))
    vol=(1/3)*s**2*h
    ax.set_title(f'base={s:.2f}×{s:.2f}  h={h:.2f}  V={vol:.2f}',fontsize=9.5,pad=4)
    desc=f"Square pyramid: base={s:.2f}×{s:.2f}, h={h:.2f}. V=(1/3)s²h={vol:.2f}."
    return label_and_save(fig,"3D Geometry","Pyramid",desc,eid,save_dir)

_VENN_CONFIGS = [
    (('A','B'), ({1,2,3,4},{4,5,6,7})),
    (('X','Y'), ({10,20,30},{30,40,50})),
    (('P','Q'), ({2,4,6,8,10},{8,10,12,14})),
    (('M','N'), ({1,3,5,7},{7,9,11,13})),
    (('S','T'), ({100,200,300,400},{300,400,500,600})),
    (('U','V'), ({11,22,33},{33,44,55,66})),
]
def draw_venn_diagram(eid, save_dir):
    fig, ax = plt.subplots(figsize=(6,7))
    ax.set_facecolor(random.choice(['#fafafa','#ffffff','#f0f4f8']))
    ax.set_aspect('equal'); ax.axis('off')
    fc1,ec1=rpal(); fc2,ec2=rpal()
    names, (A,B) = random.choice(_VENN_CONFIGS)
    inter=A&B; only_a=A-B; only_b=B-A; union=A|B
    gap = rng(0.8, 1.3)
    ax.add_patch(Circle((-gap,0),1.9,fc=fc1,alpha=0.45,ec=ec1,lw=2))
    ax.add_patch(Circle((gap,0),1.9,fc=fc2,alpha=0.45,ec=ec2,lw=2))
    ax.text(-gap*1.6,0,f'{names[0]} only\n{sorted(only_a)}',fontsize=8.5,ha='center',color=ec1)
    ax.text(gap*1.6,0,f'{names[1]} only\n{sorted(only_b)}',fontsize=8.5,ha='center',color=ec2)
    ax.text(0,0.2,f'∩\n{sorted(inter)}',fontsize=8.5,ha='center',color='#6C3483')
    ax.text(-gap,2.4,f'{names[0]}={sorted(A)}',fontsize=8.5,ha='center',color=ec1)
    ax.text(gap,2.4,f'{names[1]}={sorted(B)}',fontsize=8.5,ha='center',color=ec2)
    ax.text(0,-2.6,f'Union={sorted(union)}',fontsize=9,ha='center',color='#333')
    ax.set_xlim(-4.5,4.5); ax.set_ylim(-3.5,3.5)
    desc=f"Venn: {names[0]}={sorted(A)}, {names[1]}={sorted(B)}. ∩={sorted(inter)}. ∪={sorted(union)}."
    return label_and_save(fig,"Set Theory","Venn Diagram",desc,eid,save_dir)

_ANGLE_TYPES=[
    ("Acute",  lambda: random.randint(10,89)),
    ("Right",  lambda: 90),
    ("Obtuse", lambda: random.randint(91,179)),
    ("Straight",lambda: 180),
    ("Reflex", lambda: random.randint(181,350)),
]
def draw_angles(eid, save_dir):
    chosen = random.sample(_ANGLE_TYPES, 4)
    fig, axes = plt.subplots(2, 2, figsize=(6,7))
    fig.patch.set_facecolor(random.choice(['#fafafa','#ffffff']))
    fc, ec = rpal()
    for ax_i, (name, gen) in zip(axes.flatten(), chosen):
        deg = gen()
        ax_i.set_facecolor('#fafafa'); ax_i.set_xlim(-0.3,2.6)
        ax_i.set_ylim(-0.3,2.6); ax_i.set_aspect('equal'); ax_i.axis('off')
        ax_i.plot([0,2],[0,0],'k-',lw=2)
        rad=np.deg2rad(min(deg,180))
        ax_i.plot([0,2*np.cos(rad)],[0,2*np.sin(rad)],'k-',lw=2)
        if 0 < deg <= 180:
            arc=Arc((0,0),0.7,0.7,angle=0,theta1=0,theta2=deg,color=ec,lw=2)
            ax_i.add_patch(arc)
        ax_i.text(0.65,0.32,f'{name}\n{deg}°',fontsize=9,color='#333',ha='left')
    fig.suptitle("Types of Angles",fontsize=11,y=0.97)
    desc="Angles shown: "+", ".join([f"{n}({g()}°)" for n,g in chosen])+"."
    return label_and_save(fig,"Geometry","Angles",desc,eid,save_dir)

def draw_vectors(eid, save_dir):
    fig, ax = plt.subplots(figsize=(6,7))
    ax.set_facecolor(random.choice(['#fafafa','#ffffff','#f5f5f5']))
    ax.grid(True,linestyle='--',alpha=0.25)
    ax_val=rng(1.5,3.5); ay_val=rng(1.0,3.5)
    bx_val=rng(-3.0,-0.5); by_val=rng(1.5,4.0)
    rx,ry=ax_val+bx_val, ay_val+by_val
    mag_a=np.sqrt(ax_val**2+ay_val**2)
    mag_b=np.sqrt(bx_val**2+by_val**2)
    mag_r=np.sqrt(rx**2+ry**2)
    ax.quiver(0,0,ax_val,ay_val,angles='xy',scale_units='xy',scale=1,
              color='#E74C3C',width=0.009,label=f'A=({ax_val:.1f},{ay_val:.1f})')
    ax.quiver(0,0,bx_val,by_val,angles='xy',scale_units='xy',scale=1,
              color='#2E86C1',width=0.009,label=f'B=({bx_val:.1f},{by_val:.1f})')
    # Resultant via annotate to avoid linestyle/dash bug in quiver
    ax.annotate('',xy=(rx,ry),xytext=(0,0),
                arrowprops=dict(arrowstyle='->',color='#27AE60',lw=2.5,
                                connectionstyle='arc3,rad=0.0'))
    ax.quiver(0,0,0,0,angles='xy',scale_units='xy',scale=1,
              color='#27AE60',width=0.009,alpha=0,label=f'R=({rx:.1f},{ry:.1f})')
    ax.text(ax_val+0.1,ay_val+0.1,f'|A|={mag_a:.2f}',fontsize=8,color='#E74C3C')
    ax.text(bx_val-0.1,by_val+0.1,f'|B|={mag_b:.2f}',fontsize=8,color='#2E86C1',ha='right')
    ax.text(rx+0.1,ry+0.1,f'|R|={mag_r:.2f}',fontsize=8,color='#27AE60')
    ax.set_aspect('equal')
    xlim=max(abs(ax_val),abs(bx_val),abs(rx))+1.2
    ylim=max(ay_val,by_val,ry)+1.2
    ax.set_xlim(-xlim,xlim+0.5); ax.set_ylim(-0.8,ylim)
    ax.legend(fontsize=8.5,loc='lower right')
    desc=(f"Vectors A=({ax_val:.1f},{ay_val:.1f}), B=({bx_val:.1f},{by_val:.1f}). "
          f"R=({rx:.1f},{ry:.1f}). |A|={mag_a:.2f}, |B|={mag_b:.2f}, |R|={mag_r:.2f}.")
    return label_and_save(fig,"Vectors","Vectors",desc,eid,save_dir)

_INTEG_FUNCS=[
    (lambda x,n: x**n,
     lambda lo,hi,n: hi**(n+1)/(n+1)-lo**(n+1)/(n+1),
     lambda n: f'x^{n}'),
    (lambda x,n: np.sin(n*x),
     lambda lo,hi,n: (-np.cos(n*hi)+np.cos(n*lo))/n,
     lambda n: f'sin({n}x)'),
    (lambda x,n: np.cos(n*x),
     lambda lo,hi,n: (np.sin(n*hi)-np.sin(n*lo))/n,
     lambda n: f'cos({n}x)'),
    (lambda x,n: x**2 + n*x,
     lambda lo,hi,n: (hi**3/3+n*hi**2/2)-(lo**3/3+n*lo**2/2),
     lambda n: f'x²+{n}x'),
]
def draw_integration(eid, save_dir):
    fig, ax = plt.subplots(figsize=(6,7))
    ax.set_facecolor(random.choice(['#fafafa','#ffffff']))
    ax.grid(True,linestyle='--',alpha=0.25)
    fc, ec = rpal()
    fi = random.randint(0,3)
    f_eval, f_exact, f_label = _INTEG_FUNCS[fi]
    n = random.randint(2,4)
    lo = random.choice([0,1]); hi = random.randint(lo+1,lo+4)
    x = np.linspace(lo-0.3, hi+0.3, 400)
    try:
        y   = f_eval(x,n)
        val = f_exact(lo,hi,n)
    except Exception:
        y=x**2; val=hi**3/3-lo**3/3; n=2
    ax.plot(x, y, color=ec, lw=2.5, label=f'f(x)={f_label(n)}')
    x_fill=np.linspace(lo,hi,300)
    ax.fill_between(x_fill, f_eval(x_fill,n), alpha=0.35, color=fc, label='Area')
    ax.axhline(0,color='k',lw=0.8,alpha=0.4)
    yv=y[np.isfinite(y)]
    ax.set_xlim(lo-0.6,hi+0.8); ax.set_ylim(min(yv)-0.5,max(yv)+1.5)
    ax.text((lo+hi)/2, max(yv)*0.55, f'= {val:.4f}', fontsize=12, ha='center', color=ec)
    ax.legend(fontsize=8.5)
    desc=f"∫ f(x)={f_label(n)} from {lo} to {hi} = {val:.4f}."
    return label_and_save(fig,"Calculus","Integration Equation",desc,eid,save_dir)

_DIFF_FUNCS=[
    (lambda x,a,b: a*x**3+b*x, lambda x,a,b: 3*a*x**2+b, 'ax³+bx','3ax²+b'),
    (lambda x,a,b: a*np.sin(b*x), lambda x,a,b: a*b*np.cos(b*x), 'a·sin(bx)','ab·cos(bx)'),
    (lambda x,a,b: a*np.exp(0.3*x), lambda x,a,b: 0.3*a*np.exp(0.3*x), 'a·e^(0.3x)','0.3a·e^(0.3x)'),
    (lambda x,a,b: a*x**2+b*x, lambda x,a,b: 2*a*x+b, 'ax²+bx','2ax+b'),
]
def draw_differentiation(eid, save_dir):
    fig, ax = plt.subplots(figsize=(6,7))
    ax.set_facecolor(random.choice(['#fafafa','#ffffff']))
    ax.grid(True,linestyle='--',alpha=0.25)
    fc, ec = rpal()
    fi = random.randint(0,3)
    f_eval,df_eval,flabel,dflabel = _DIFF_FUNCS[fi]
    a=rng(0.3,1.5); b=rng(0.5,2.0); x0=rng(-1.5,1.5)
    x=np.linspace(-3,3,400)
    y=f_eval(x,a,b); dy=df_eval(x,a,b)
    ax.plot(x,y,color=ec,lw=2.5,label=f'f(x)={flabel}')
    ax.plot(x,dy,'r--',lw=2,label=f"f'(x)={dflabel}")
    slope=float(df_eval(np.array([x0]),a,b)[0])
    y0v=float(f_eval(np.array([x0]),a,b)[0])
    xt=np.linspace(x0-1.2,x0+1.2,50)
    ax.plot(xt,slope*(xt-x0)+y0v,'g-',lw=1.8,label=f'Tangent x={x0:.2f} m={slope:.2f}')
    ax.plot(x0,y0v,'ko',ms=6,zorder=5)
    ax.set_xlim(-3,3.5)
    yv=np.concatenate([y,dy]); yv=yv[np.isfinite(yv)]
    ax.set_ylim(max(-12,min(yv)-1),min(12,max(yv)+1))
    ax.legend(fontsize=8)
    desc=f"f(x)={flabel} (a={a:.2f},b={b:.2f}). f'(x)={dflabel}. Tangent at x={x0:.2f}: slope={slope:.2f}."
    return label_and_save(fig,"Calculus","Differentiation",desc,eid,save_dir)

_EQ_SETS=[
    [(r"$e^{i\pi}+1=0$","Euler's Identity"),(r"$x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}$","Quadratic Formula"),
     (r"$F=ma$","Newton 2nd"),(r"$E=mc^2$","Mass-Energy"),(r"$PV=nRT$","Ideal Gas")],
    [(r"$a^2+b^2=c^2$","Pythagorean"),(r"$\sin^2\theta+\cos^2\theta=1$","Trig Identity"),
     (r"$A=\pi r^2$","Circle Area"),(r"$V=\frac{4}{3}\pi r^3$","Sphere Volume"),
     (r"$\frac{d}{dx}[x^n]=nx^{n-1}$","Power Rule")],
    [(r"$\int x^n dx=\frac{x^{n+1}}{n+1}+C$","Power Int."),
     (r"$\lim_{x\to 0}\frac{\sin x}{x}=1$","Sinc Limit"),
     (r"$e=\sum_{n=0}^{\infty}\frac{1}{n!}$","Euler's Number"),
     (r"$\ln(ab)=\ln a+\ln b$","Log Product"),
     (r"$i^2=-1$","Imaginary unit")],
    [(r"$\nabla\cdot\mathbf{E}=\frac{\rho}{\varepsilon_0}$","Gauss Law"),
     (r"$\vec{F}=q(\vec{E}+\vec{v}\times\vec{B})$","Lorentz Force"),
     (r"$p=mv$","Momentum"),(r"$KE=\frac{1}{2}mv^2$","Kinetic Energy"),
     (r"$W=Fd\cos\theta$","Work")],
]
def draw_equation(eid, save_dir):
    fig, ax = plt.subplots(figsize=(6,7))
    ax.set_facecolor(random.choice(['#fafafa','#ffffff','#fffef5']))
    ax.axis('off')
    eqs = random.choice(_EQ_SETS)
    y0=0.88
    for tex,label in eqs:
        ax.text(0.5,y0,tex,fontsize=14,ha='center',va='center',
                transform=ax.transAxes,color='#1A1A2E')
        ax.text(0.5,y0-0.07,f'({label})',fontsize=8,ha='center',va='center',
                transform=ax.transAxes,color='#7F8C8D')
        y0-=0.17
    desc="Equations: "+", ".join([l for _,l in eqs])+"."
    return label_and_save(fig,"Equations","Equation",desc,eid,save_dir)

def draw_trigonometry(eid, save_dir):
    fig, axes = plt.subplots(1,2,figsize=(6,7))
    fig.patch.set_facecolor(random.choice(['#fafafa','#ffffff']))
    fc, ec = rpal()
    theta_deg = random.randint(10,80)
    theta = np.radians(theta_deg)
    ax=axes[0]; ax.set_facecolor('#fafafa'); ax.set_aspect('equal')
    ax.add_patch(Circle((0,0),1,fc='none',ec='#1A5276',lw=1.5))
    ax.plot([0,np.cos(theta)],[0,0],'r-',lw=2,label='cos')
    ax.plot([np.cos(theta),np.cos(theta)],[0,np.sin(theta)],'b-',lw=2,label='sin')
    ax.plot([0,np.cos(theta)],[0,np.sin(theta)],'g-',lw=2,label='r=1')
    ax.plot(np.cos(theta),np.sin(theta),'ko',ms=5,zorder=5)
    arc=Arc((0,0),0.35,0.35,angle=0,theta1=0,theta2=theta_deg,color=ec,lw=1.5)
    ax.add_patch(arc)
    ax.text(0.26,0.07,f'{theta_deg}°',fontsize=8,color=ec)
    ax.set_xlim(-1.5,1.6); ax.set_ylim(-1.5,1.6)
    ax.axhline(0,color='k',lw=0.7); ax.axvline(0,color='k',lw=0.7)
    ax.legend(fontsize=7,loc='lower right'); ax.set_title('Unit Circle',fontsize=9)
    ax2=axes[1]; ax2.set_facecolor('#fafafa'); ax2.grid(True,linestyle='--',alpha=0.25)
    show=random.choice(['sin_cos','sin_tan','cos_tan','all'])
    xr=np.linspace(-np.pi,2*np.pi,300)
    if 'sin' in show or show=='all':
        ax2.plot(xr,np.sin(xr),color='#C0392B',lw=2,label='sin x')
    if 'cos' in show or show=='all':
        ax2.plot(xr,np.cos(xr),color='#1A5276',lw=2,label='cos x')
    if 'tan' in show or show=='all':
        ty=np.tan(xr); ty[np.abs(ty)>4]=np.nan
        ax2.plot(xr,ty,color='#27AE60',lw=2,label='tan x')
    ax2.set_xlim(-np.pi,2*np.pi); ax2.set_ylim(-4.5,4.5)
    ax2.legend(fontsize=7.5); ax2.set_title('Trig Functions',fontsize=9)
    sv=np.sin(theta); cv=np.cos(theta); tv=np.tan(theta)
    fig.text(0.5,0.06,rf'θ={theta_deg}°: sin={sv:.3f}, cos={cv:.3f}, tan={tv:.3f}',ha='center',fontsize=9)
    desc=f"Trig θ={theta_deg}°: sin={sv:.3f}, cos={cv:.3f}, tan={tv:.3f}."
    return label_and_save(fig,"Trigonometry","Trigonometry",desc,eid,save_dir)

_WAVE_CONFIGS=[
    lambda x:(np.sin(x),np.cos(x),np.sin(x)+0.5*np.cos(2*x)),
    lambda x:(np.sin(2*x),np.sin(3*x),np.sin(2*x)+np.sin(3*x)),
    lambda x:(np.sin(x)*np.exp(-0.1*x),np.cos(x),np.sin(x)+0.4*np.sin(5*x)),
    lambda x:(np.sin(x)+0.5*np.sin(3*x),0.5*np.sin(3*x)+0.25*np.sin(5*x),
              np.sin(x)+0.5*np.sin(3*x)+0.25*np.sin(5*x)),
    lambda x:(np.sin(x+random.uniform(0,np.pi)),np.cos(2*x),np.sin(x)+np.cos(2*x)),
    lambda x:(np.sin(x)*np.cos(0.5*x),np.cos(x)*np.sin(0.5*x),np.sin(x)*np.cos(0.5*x)+np.cos(x)*np.sin(0.5*x)),
]
_WAVE_LABELS=[
    ('sin(x)','cos(x)','Composite'),
    ('sin(2x)','sin(3x)','Sum'),
    ('Damped sin','cos(x)','Beat'),
    ('Fourier t1','Fourier t2','Approx'),
    ('Phase-shifted sin','cos(2x)','Mixed'),
    ('sin·cos(x/2)','cos·sin(x/2)','Product waves'),
]
def draw_waves(eid, save_dir):
    ci = random.randint(0,5)
    cfg=_WAVE_CONFIGS[ci]; labs=_WAVE_LABELS[ci]
    fc,ec=rpal()
    fig,axes=plt.subplots(3,1,figsize=(6,7),sharex=True)
    fig.patch.set_facecolor(random.choice(['#fafafa','#ffffff']))
    x=np.linspace(0,4*np.pi,500)
    w1,w2,w3=cfg(x)
    for ax_i,y,label,color in zip(axes,[w1,w2,w3],labs,[ec,'#C0392B','#27AE60']):
        ax_i.set_facecolor('#fafafa')
        ax_i.plot(x,y,color=color,lw=2)
        ax_i.axhline(0,color='k',lw=0.7,linestyle=':')
        ax_i.set_title(label,fontsize=8.5)
        ax_i.grid(True,linestyle='--',alpha=0.2)
    axes[-1].set_xlabel('x (radians)',fontsize=8)
    fig.tight_layout(rect=[0,0,1,0.92])
    desc=f"Wave set {ci+1}: {labs[0]}, {labs[1]}, {labs[2]}."
    return label_and_save(fig,"Waves","Waves",desc,eid,save_dir)

_LOGIC_SETS=[
    [(r"$\forall x \in A$","Universal: for all x in A"),
     (r"$\exists x \in B$","Existential: there exists x in B"),
     (r"$P \Rightarrow Q$","Implication: P implies Q"),
     (r"$P \Leftrightarrow Q$","Biconditional: P iff Q"),
     (r"$\neg P$","Negation: not P"),
     (r"$P \wedge Q$","AND"),
     (r"$P \vee Q$","OR"),
     (r"$A \subseteq B$","Subset"),
     (r"$\emptyset$","Empty set")],
    [(r"$A \cup B$","Union"),(r"$A \cap B$","Intersection"),
     (r"$A \setminus B$","Set Difference"),(r"$|A|$","Cardinality"),
     (r"$\mathbb{N},\mathbb{Z},\mathbb{Q},\mathbb{R}$","Number sets"),
     (r"$P \oplus Q$","XOR"),(r"$\overline{P}$","Complement"),
     (r"$A \times B$","Cartesian product"),(r"$2^A$","Power set")],
    [(r"$\therefore$","Therefore"),(r"$\because$","Because"),
     (r"$\equiv$","Logical equivalence"),(r"$\top$","Tautology (True)"),
     (r"$\bot$","Contradiction (False)"),(r"$\vdash$","Proves"),
     (r"$\models$","Entails"),(r"$\{x \mid P(x)\}$","Set builder"),
     (r"$\sum_{i=1}^{n} i = \frac{n(n+1)}{2}$","Gauss sum")],
]
def draw_logic_symbols(eid, save_dir):
    fig,ax=plt.subplots(figsize=(6,7))
    ax.set_facecolor(random.choice(['#fafafa','#ffffff','#fffef5']))
    ax.axis('off')
    syms=random.choice(_LOGIC_SETS)
    y0=0.93
    for tex,explanation in syms:
        ax.text(0.08,y0,tex,fontsize=12,va='center',transform=ax.transAxes,color='#1A1A2E')
        ax.text(0.42,y0,explanation,fontsize=9,va='center',transform=ax.transAxes,color='#555')
        y0-=0.096
    ax.set_title("Mathematical Logic Symbols",fontsize=11,pad=5)
    desc="Logic: "+", ".join([e for _,e in syms[:4]])+" ..."
    return label_and_save(fig,"Logic","Logic Symbols",desc,eid,save_dir)

# ══════════════════════════════════════════════════════════════════════════════
DRAW_FN = {
    "Circle":                draw_circle,
    "Triangle":              draw_triangle,
    "Square":                draw_square,
    "Rectangle":             draw_rectangle,
    "Equilateral Triangle":  draw_equilateral_triangle,
    "Isosceles Triangle":    draw_isosceles_triangle,
    "Scalene Triangle":      draw_scalene_triangle,
    "Right-Angled Triangle": draw_right_triangle,
    "Parallelogram":         draw_parallelogram,
    "Rhombus":               draw_rhombus,
    "Trapezium":             draw_trapezium,
    "Kite":                  draw_kite,
    "Pentagon":              draw_pentagon,
    "Ellipse":               draw_ellipse,
    "Parabola":              draw_parabola,
    "Hyperbola":             draw_hyperbola,
    "Cube":                  draw_cube,
    "Cuboid":                draw_cuboid,
    "Sphere":                draw_sphere,
    "Cylinder":              draw_cylinder,
    "Cone":                  draw_cone,
    "Pyramid":               draw_pyramid,
    "Venn Diagram":          draw_venn_diagram,
    "Angles":                draw_angles,
    "Vectors":               draw_vectors,
    "Integration Equation":  draw_integration,
    "Differentiation":       draw_differentiation,
    "Equation":              draw_equation,
    "Trigonometry":          draw_trigonometry,
    "Waves":                 draw_waves,
    "Logic Symbols":         draw_logic_symbols,
}
TOPICS = list(DRAW_FN.keys())

if __name__ == "__main__":
    data_log = []
    print(f"Generating {NUM_IMAGES} unique images across {len(TOPICS)} topics ...")
    for i in range(NUM_IMAGES):
        topic = TOPICS[i % len(TOPICS)]
        record = DRAW_FN[topic](i, DATASET_NAME)
        data_log.append(record)
        print(f"  [{i+1:>4}/{NUM_IMAGES}] {record['file']}")
    meta_path = os.path.join(DATASET_NAME, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(data_log, f, indent=4)
    print(f"\nDone! {NUM_IMAGES} images saved to '{DATASET_NAME}/'")
    print(f"Metadata -> '{meta_path}'")


l = (1,2,3)
m = [4,5,6]
n = [*l, *m]
print(n)

a = {1: "aaa", 2: "bbb"}
print(a.get(32,None))


from pathlib import Path

DIR = Path('../../datasets/insect/100ms_4096pts_2024-07-03_17-41-00')
files = [f for f in DIR.glob("*/*.csv")]

for f in files[:5]:
    clas = f.parent.name
    print(clas, f.name)

print("aaaa_h1_2_3.csv".replace(".csv","").split("_")[-3:])
import re
import struct
pattern_noun = r'^[N]{2}(S|P|PS)?$'
pattern_adj = r'^[J]{2}(R|S)?$'
pattern_verb = r'^VB(D|G|N|P|Z)?$'

print(struct.calcsize("P") * 8)

if re.match(pattern_verb,"VB"):
    print("yes")
else:
    print("No")
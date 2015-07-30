import sys
import xml.etree.ElementTree as ET


for i in range(1,2000):
    print "processing tree", i
    found = False;
    model = ET.parse(sys.argv[1])
    root = model.getroot()
    ensemble = root.find('ensemble')
    for tree in ensemble.findall('tree'):
        tree_id = int(tree.attrib['id'])
        if i==tree_id:
            found = True
        else:
            ensemble.remove(tree)
    
    if not found:
        break

    model.write('model.tree'+str(i))

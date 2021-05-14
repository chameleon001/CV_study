#%%
public_data_path = "../../data"
import pydot

from PIL import Image

#%%

g = pydot.Dot(graph_type='graph')

g.add_node(pydot.Node(str(0), fontcolor='transparent'))

for i in range(5):
    g.add_node(pydot.Node(str(i+1)))
    g.add_node(pydot.Edge(str(0), str(i+1)))

    for j in range(5):
        g.add_node(pydot.Node(str(j+1)+'-'+str(i+1)))
        g.add_node(pydot.Edge(str(j+1)+'-'+str(i+1),str(j+1)))

g.write_png('graph.jpg',prog='neato')
# %%

threshold = 2

path = public_data_path
g = pydot.Dot(graph_type = 'graph')

for i in range(nbr_images):
    for j in range(i+1, nbr_images):
        if matchscores[i,j] > threshold:

            #first image in pair
            im = Image.open(imlist[i])
            im.thumvnail((100,100))
            #resize와 비슷함 썸네일로 만들어줌
            filename = str(i) + '.png'
            im.save(filename)
            # need temporary file of the right size
            g.add_node(pydot.Node(str(i), fontcolor='transparent', shape='rectangle', image = path+filename))

            g.add_edge(pydot.Edge(str(i),str(j)))

g.write_png('whitehouse.png')
import json

programs =  { "p1" : 64,
              "p2" : 64, 
              "p3" : 30,
              "p4" : 4,
              "p5" : 5 }

def save_setup(system_setup):
    data_setup = json.dumps(system_setup)
    f2 = open('data/programs/program_base.json', 'w')
    f2.write(data_setup)
    f2.close()

def load_setup():

    f = open('data/programs/program_base.json' ,'r')
    content = f.read() # open the setup file
    settings = json.loads(content) 
    f.close() # close the setup file
    return settings

save_setup(programs)
print(load_setup())
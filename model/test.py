from model.initial_conditions import IC
import torch
import model.utilities as util
def test(model_file,config_file):
    

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = util.read_configs(config_file)

    model = torch.load(model_file)

    inputs,outputs = util.model_inference(config['spatial_grid'],0.5,model)
    util.tensor_to_file(inputs,'inputs.txt')
    
    util.visualize_model_output(inputs,outputs,2)



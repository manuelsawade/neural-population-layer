from torch import nn

from activations.neuron import NeuronPopulation

def activation_metric(model: nn.Module, append_to: dict = {}) -> None:

    def get_activation(name, type):
        def hook(module, input, output):                
            if 'scores' not in append_to:
                append_to['scores'] = {}

            if 'output' not in append_to:
                append_to['output'] = {}
            
            if isinstance(module, NeuronPopulation):
                if not module.pop_out == None:
                    inter_name = f'{name}.inter'
                    if inter_name not in append_to['output']:
                        append_to['output'][inter_name] = {}
                        append_to['output'][inter_name]['values'] = []
                        append_to['output'][inter_name]['type'] = f'{type}.inter'

                    append_to['output'][inter_name]['values'].extend(module.pop_out.tolist())
            
            if name not in append_to['scores']:
                append_to['scores'][name] = {}
                append_to['scores'][name]['mean'] = 0
                append_to['scores'][name]['std'] = []
                append_to['scores'][name]['len'] = []
            
            append_to['scores'][name]['mean'] += output.detach().mean()
            append_to['scores'][name]['std'].append(output.detach().std())
            append_to['scores'][name]['len'].append(len(output.detach()))

            if name not in append_to['output']:
                append_to['output'][name] = {}
                append_to['output'][name]['type'] = type
                append_to['output'][name]['values'] = []
            
            append_to['output'][name]['values'].extend(output.detach().tolist())
        return hook
    
    for index in range(len(model.layers)):
        model.layers[index].register_forward_hook(get_activation(f'layers.{index}', model.layers[index]._get_name()))

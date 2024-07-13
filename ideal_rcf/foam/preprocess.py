from ideal_rcf.dataloader.caseset import CaseSet

from typing import Optional, Dict
from pathlib import Path
import os

class FoamParser(object):
    '''
    2 utilizatons:
    1) Receives an anisotropy field (pre realizable) and extracts the eV from avg eV from both folds
       Outputs the NL part by substractin to a the eV*Shat term and writes it to semi_impliciti
       
       
    2) Receives the realizable anisotropy field and writes it to explicit
    '''
    def __init__(self, 
                 caseset :CaseSet,
                 pass_iterations_dict :Optional[Dict]=None,
                 pass_boundaries_dict :Optional[Dict]=None,
                 pass_fixed_value_dict :Optional[Dict]=None):
        
        if not isinstance(caseset, CaseSet):
            raise AssertionError(f'[config_error] base_model_config must be of instance {CaseSet()}')
        
        self.caseset = caseset
                
        self.iterations = {
            'BUMP': 6000,
            'CNDV': 5000,
            'PHLL': 20000
        }
        if pass_iterations_dict:
            self.iterations.update(pass_iterations_dict)
        
        self.boundaries_dict = {
            'top_bottom': {
                'PHLL': 'Wall',
                'BUMP': '',
                'CNDV': ''  
            },
            'empty': {
                'PHLL': 'defaultFaces',
                'BUMP': 'frontAndBack',
                'CNDV': 'frontAndBack'  
            },
            'inlet_outlet': {
                'PHLL': 'cyclic',
                'BUMP': 'zeroGradient',
                'CNDV': 'zeroGradient'
            }
        } 
        if pass_boundaries_dict:
            self.boundaries_dict.update(pass_boundaries_dict)
        
        self.fixed_value_dict = {
            'a' : '(0 0 0 0 0 0)',
            'nut': '0'
        }
        if pass_fixed_value_dict:
            self.fixed_value_dict.update(pass_fixed_value_dict)


    def foam_header(self, 
                    field_type :str,
                    iterations :str, 
                    field :str):
        return f'''/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\    /   O peration     | Version:  2006                                  |
|   \\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\/     M anipulation  |                                                 |
\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       {field_type};
    location    "{iterations}";
    object      {field};
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //'''


    def create_boundaries(self,
                          _id :str):
        return f'''
boundaryField
{{
    inlet
    {{
        type            {self.boundaries_dict['inlet_outlet'][self.caseset.case[0][:4]]};
    }}
    outlet
    {{
        type            {self.boundaries_dict['inlet_outlet'][self.caseset.case[0][:4]]};
    }} 
    top{self.boundaries_dict['top_bottom'][self.caseset.case[0][:4]]}
    {{
        type            fixedValue;
        value           uniform {self.fixed_value_dict[_id]};
    }}
    bottom{self.boundaries_dict['top_bottom'][self.caseset.case[0][:4]]}
    {{
        type            fixedValue;
        value           uniform {self.fixed_value_dict[_id]};
    }}
    {self.boundaries_dict['empty'][self.caseset.case[0][:4]]}
    {{
        type            empty;
    }}
}}
'''


    def create_anisotropy(self):
        '''
        (
        (xx xy xz yy yz zz)
        ...
        )
        '''
        anisotropy_header = f'''
        
dimensions      [0 2 -2 0 0 0 0];


internalField   nonuniform List<symmTensor>
{self.caseset.predictions.shape[0]}
'''
        anisotropy_reg = '(\n'+'\n'.join([f'({anisotropy[0]} {anisotropy[1]} 0 {anisotropy[2]} 0  {anisotropy[3]})' for anisotropy in self.caseset.predictions])+'\n)\n;'
        ### do I need to add boundary field?
           
        anisotropy_bottom = '''\n\n// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //'''
        return ''.join([anisotropy_header, anisotropy_reg, self.create_boundaries('a'), anisotropy_bottom])


    def create_viscosity(self,
                         implicit :Optional[bool]=True):
        viscosity_header = f'''
        
dimensions      [0 2 -1 0 0 0 0];


internalField   nonuniform List<scalar>
{self.caseset.predictions_oev.shape[0]}
'''
        viscosity_reg = '(\n'+'\n'.join([f'{viscosity if implicit else 0}' for viscosity in self.caseset.predictions_oev])+'\n)\n;'
        ### do I need to add boundary field?
        viscosity_bottom = '''\n\n// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //'''
        return ''.join([viscosity_header, viscosity_reg, self.create_boundaries('nut'), viscosity_bottom])


    def dump_predictions(self,
                         dir_path :Path):
        foam_dir = f'{dir_path}/foam/{self.caseset.case[0]}'
        if not os.path.exists(foam_dir):
            os.makedirs(foam_dir)
        
        try:
            bool(self.caseset.predictions)
            raise ValueError('Make sure inference has been ran')
        
        except ValueError:
            foam_predictions = f'{foam_dir}/predicitons'
            foam_results= f'{foam_dir}/results'
            if not os.path.exists(foam_predictions):
                os.mkdir(foam_predictions)
            if not os.path.exists(foam_results):
                os.mkdir(foam_results)

            with open(f'{foam_predictions}/aperp', 'w') as _file:
                _file.write(''.join([
                        self.foam_header('volSymmTensorField', 
                                         self.iterations[self.caseset.case[0][:4]], 
                                         'aperp'),
                        self.create_anisotropy()
                    ])
                )
            print(f'> dumped {foam_predictions}/aperp')

            try:
                bool(self.caseset.predictions_oev)
                implicit = False            
            except ValueError:
                implicit = True

            with open(f'{foam_predictions}/nut_L', 'w') as _file:
                _file.write(''.join([
                        self.foam_header('volScalarField', 
                                        self.iterations[self.caseset.case[0][:4]], 
                                        'nut_L'),
                        self.create_viscosity(implicit=implicit)
                    ])
                )
            print(f'> dumped {foam_predictions}/nut_L')
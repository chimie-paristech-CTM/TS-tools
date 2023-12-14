import os
import shutil

def copy_final_outputs(dir_name):
    os.makedirs('final_outputs', exist_ok=True)
    for reaction_dir in os.listdir(dir_name):
        print(reaction_dir)
        final_ts_guess_dir = os.path.join(os.path.join(dir_name, reaction_dir), 'final_ts_guess')
        if len(os.listdir(final_ts_guess_dir)) != 0:
            shutil.copytree(final_ts_guess_dir, os.path.join('final_outputs', f'final_outputs_{reaction_dir}'))
            shutil.copy(os.path.join(os.path.join(dir_name, reaction_dir), 'rp_geometries/reactants_geometry.xyz'),
                        os.path.join('final_outputs', f'final_outputs_{reaction_dir}/'))
            shutil.copy(os.path.join(os.path.join(dir_name, reaction_dir), 'rp_geometries/products_geometry.xyz'),
                        os.path.join('final_outputs', f'final_outputs_{reaction_dir}/'))
            
if __name__ == '__main__':
    copy_final_outputs('benchmarking_150')

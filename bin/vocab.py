import pandas as pd
from itertools import combinations_with_replacement

def generate_combinations(base_symbol_list, add_cross_interaction=False):
    combinations = list(combinations_with_replacement( base_symbol_list, 2))
    all_combinations = combinations
    if(add_cross_interaction == True):
        cross_interaction = list(combinations_with_replacement([x[0]+x[1] for x in combinations], 2))
        all_combinations += cross_interaction
    return all_combinations
    
def new_amino_acid_vocab():
    dl_flags = ['[PAD]','[CLS]','[MASK]']
    amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 
                   'P', 'Q','R', 'S', 'T', 'V', 'W', 'Y']
    chemical = ['AL', 'SU', 'AC', 'AC', 'AR', 'AL', 'BA', 'AL', 'BA',
                'AL', 'SU', 'AM', 'AL', 'AM', 'BA', 'HY', 'HY', 'AL', 
                'AR', 'AR']
    charge = ['UC', 'UC', 'NC', 'NC', 'UC', 'UC', 'PC', 'UC', 'PC', 'UC', 'UC', 'UC', 'UC',
              'UC', 'PC', 'UC', 'UC', 'UC', 'UC', 'UC']
    H_donor = ['HN', 'HN', 'HA', 'HA', 'HN', 'HN', 'HX', 'HN', 'HD', 'HN', 'HN', 'HX',
               'HN', 'HX', 'HD', 'HX', 'HX', 'HN', 'HD', 'HX']
    polarity = ['PN', 'PN', 'PP', 'PP', 'PN', 'PN', 'PP', 'PN', 'PP', 'PN', 'PN', 'PP',
                'PN', 'PP', 'PP', 'PP', 'PP', 'PN', 'PN', 'PP']
    hydropathy = ['OH', 'OH', 'OP', 'OP', 'OH', 'ON', 'ON', 'OH', 'OP', 'OH', 'OH', 'OP',
                   'ON', 'OP', 'OP', 'ON', 'ON', 'OH', 'OH', 'ON']
    
    volume = ['VVS', 'VS', 'VS', 'VM', 'VVL', 'VVS', 'VM', 'VL', 'VL', 'VL', 'VL', 'VS',
               'VS', 'VM', 'VL', 'VVS', 'VS', 'VM', 'VVL', 'VVL']
    
    chemical_classes = ['AL', 'AR', 'BA', 'AC', 'AM', 'SU', 'HY']
    charge_classes = ['UC', 'PC', 'NC']
    H_donor_classes = ['HN', 'HA', 'HD', 'HX']
    polarity_classes = ['PP', 'PN']
    hydropathy_classes = ['OH', 'ON', 'OP']
    volume_classes = ['VVS', 'VS', 'VM', 'VL', 'VVL']
    
    amino_vocab = dl_flags + generate_combinations(amino_acids)
    addtional_names = ['chemical', 'charge', 'H_donor', 'polarity', 'hydropathy', 'volume']
    addtional_vocab = [dl_flags + generate_combinations(chemical_classes), 
                       dl_flags + generate_combinations(charge_classes), 
                       dl_flags + generate_combinations(H_donor_classes), 
                       dl_flags + generate_combinations(polarity_classes), 
                       dl_flags + generate_combinations(hydropathy_classes),
                       dl_flags + generate_combinations(volume_classes)]
    
    amino_info = pd.DataFrame(zip(amino_acids, chemical,charge, H_donor, polarity,
                               hydropathy, volume), columns=['amino_acids', 
                                                             'chemical', 'charge', 
                                                             'H_donor', 'polarity', 
                                                             'hydropathy', 'volume']).set_index('amino_acids')
    mvocab = dict([(x,i) for i,x in enumerate(amino_vocab)])
    ovocab = dict([(addtional_names[vidx], dict([(x,i) for i,x in enumerate(vocab_info)])) for vidx, vocab_info in enumerate(addtional_vocab)])
    return mvocab, ovocab, amino_info
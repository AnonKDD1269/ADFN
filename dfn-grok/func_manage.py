import types



def transfer_methods(source_classes, target_class):
    for source_class in source_classes:
        f_list = [dir(source_class)[i] for i in range(len(dir(source_class))) if not dir(source_class)[i].startswith('_')]
        for method_name in f_list:
            transfer_method(source_class, target_class, method_name)



def transfer_method(source_class, target_class, method_name):
    source_method = getattr(source_class, method_name)
    transferred_method = types.MethodType(source_method, target_class)
    setattr(target_class, method_name, transferred_method)



    
    
    
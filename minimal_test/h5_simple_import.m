%{
Import an HDF5 file to an easy navigable structure. Only Datasets,
Attributes and Groups contained in the HDF5 file will be imported, not 
Links. All HDF5 Datasets will be read completely into memory.

Dennis van Gils
29-04-2019
%}

function h5_struct = h5_simple_import(filename, varargin)
  %{
  Will import all Datasets, Attributes and Groups contained inside the HDF5
  file pointed to by 'filename'. Both main level and all nested levels will
  be imported. The output structure 'h5_struct' will mimic the HDF5
  structure tree and allows for easy navigation to the different Datasets,
  Attributes and Groups. All HDF5 Datasets will be read completely into
  memory.

  Args:
    filename (string): Location on disk pointing to the HDF5 file.

    f_verbose (boolean, optional, default=false): When true will output
      the names of the imported HDF5 structure tree to the terminal.

  Returns:
    h5_struct (struct): A structure mimicking the HDF5 structure tree.
  %}
  h5_info = h5info(filename);
  
  if nargin < 2
    f_verbose = false;
  else
    f_verbose = logical(varargin{1});
  end
    
  h5_struct = struct();
  h5_struct.filename = filename;
  h5_struct = h5_import_level(h5_struct, h5_info, f_verbose);
end

function h5_struct = h5_import_level(h5_struct, h5_group, varargin)
  %{
  Will import all Datasets, Attributes and Groups contained inside the HDF5
  structure pointed to by 'h5_group'. Lower-level (i.e. nested) groups will
  also be imported, because this function is self-recursive. The output
  structure 'h5_struct' will mimic the HDF5 structure tree and allows for
  easy navigation to the different Datasets, Attributes and Groups.

  Args:
    h5_struct (struct): A structure that will grow in place, mimicking the
      HDF5 structure tree referred to by 'h5_group'.
      NOTE: Must contain the field 'filename' at its top level pointing to
      the HDF5 file on disk.

    h5_group (struct): The HDF5 group to import. Can be the top level
      structure as returned by 'h5_info()' or any other structure referring
      to a sub-level HDF5 group.

    f_verbose (boolean, optional, default=false): When true will output
      the names of the imported HDF5 structure tree to the terminal.

  Returns:
    h5_struct (struct): A structure that will grow in place, mimicking the
      HDF5 structure tree referred to by 'h5_group'.
  %}
  if nargin < 3
    f_verbose = false;
  else
    f_verbose = logical(varargin{1});
  end

  h5_groups   = h5_group.Groups;
  h5_datasets = h5_group.Datasets;
  h5_attrs    = h5_group.Attributes;
  
  if f_verbose
    fprintf('%s\n', h5_group.Name)
  end
  
  % Cell array of strings denoting the current position inside the HDF5
  % navigation tree
  str_tree_nav = strsplit(h5_group.Name, '/');
  str_tree_nav = str_tree_nav(~cellfun('isempty', str_tree_nav));
  for i = 1:length(str_tree_nav)
      str_tree_nav{i} = matlab.lang.makeValidName(str_tree_nav{i});
  end
  
  for i_grp = 1:length(h5_datasets);
    % Import HDF5 Datasets
    str_name = h5_datasets(i_grp).Name;
    if f_verbose
      fprintf('    dset: %s\n', str_name);
    end
    
    str_fields = str_tree_nav;
    str_fields{end + 1} = matlab.lang.makeValidName(str_name);              %#ok<*AGROW>
    
    data = h5read(h5_struct.filename, ...
                  strcat(h5_group.Name, '/', str_name));
    h5_struct = setfield(h5_struct, str_fields{:}, data);
  end
  
  for i_grp = 1:length(h5_attrs);
    % Import HDF5 Attributes
    str_name = h5_attrs(i_grp).Name;
    if f_verbose
      fprintf('    attr: %s\n', str_name);
    end
    
    str_fields = str_tree_nav;
    str_fields{end + 1} = matlab.lang.makeValidName(str_name);
    
    attr = h5readatt(h5_struct.filename, h5_group.Name, str_name);
    h5_struct = setfield(h5_struct, str_fields{:}, attr);
  end
  
  for i_grp = 1:length(h5_groups);
    % Recurse into HDF5 sub-groups
    h5_struct = h5_import_level(h5_struct, ...
                                h5_groups(i_grp), ...
                                f_verbose);
  end
end


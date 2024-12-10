from stem.control import Controller
# reference: http://stackoverflow.com/questions/16977923/automatically-generate-list-of-tor-exit-nodes

exit_nodes = []

with Controller.from_port(port = 9151) as controller:
  controller.authenticate()
  # traverse all stem descriptors
  for desc in controller.get_server_descriptors():
    # check if this is a exit node
    if desc.exit_policy.is_exiting_allowed():
      # find the country that owns this exit node ip address
      locale = controller.get_info("ip-to-country/%s" % desc.address, 'unknown') if desc else 'unknown'      
      # add to list if haven't
      if (locale in exit_nodes):
        continue
      else: 
        exit_nodes += [locale]

# output countries with exit node
print exit_nodes

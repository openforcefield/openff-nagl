{% block title -%}

.. raw:: html

   <div style="display: None;">

{{ ("``" ~ objname ~ "``") | underline('=')}}

.. raw:: html

   </div>

{%- endblock %}
{% block base %}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :member-order: alphabetical  {# For consistency with Autosummary #}
   {% if show_inherited_members %}:inherited-members:
   {% endif %}{% if show_undoc_members %}:undoc-members:
   {% endif %}{% if show_inheritance %}:show-inheritance:
   {% endif %}

   {% block methods %}

   {% if methods %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
      :nosignatures:
   {% for item in methods %}
   {%- if item not in ["__new__", "__init__"] %}
      {% if show_inherited_members or item not in inherited_members -%}
         ~{{ name }}.{{ item }}
      {%- endif %}
   {% endif -%}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: {{ _('Attributes') }}

   .. autosummary::
      :nosignatures:
   {% for item in attributes %}
      {% if show_inherited_members or item not in inherited_members -%}
         ~{{ name }}.{{ item }}
      {%- endif %}
   {%- endfor %}
   {% endif %}
   {% endblock %}
{% endblock %}

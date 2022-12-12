{{ ("``" ~ objname ~ "``") | underline}}

.. currentmodule:: {{ module }}

.. auto{{ objtype }}:: {{ objname }}
    {% if objtype in ["attribute", "data"] -%}
    :no-value:
    {%- endif %}

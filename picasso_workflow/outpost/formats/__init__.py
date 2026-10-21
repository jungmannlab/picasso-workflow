"""File-format conversion glue (Phase 2 restructure).

* ``zeiss`` -- Zeiss ``.czi`` -> picasso ``.raw`` conversion
  (``convert_zeiss_file``), via picasso's native ``io.load_czi`` / ``io.save_raw``
  (the ``.czi`` reader needs the optional ``[formats]`` extra). `keep`.
"""

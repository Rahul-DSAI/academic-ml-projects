# Dataset placement

The source archive used for this project contained four folders with 1,008 PNG images each:

```text
COVID-19/
├── COVID/
├── Lung_Opacity/
├── Normal/
└── Viral Pneumonia/
```

The image files are intentionally not redistributed in this source-code package. Obtain the chest X-ray data from its original provider, check the dataset's current licence/access terms, and place the four class folders beneath a local data directory, for example `data/raw/`.

Then run:

```bash
python -m covid_xray.train --data-dir data/raw
```

Do not commit patient-identifiable or restricted medical data. Confirm that you are permitted to redistribute any dataset or derived assets before making a repository public.

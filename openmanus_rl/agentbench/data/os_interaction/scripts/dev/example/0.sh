echo '#!/bin/bash
python3 -c "print(\"%.6f\"%($*))"' > calc
chmod +x calc
mv calc /usr/local/bin/

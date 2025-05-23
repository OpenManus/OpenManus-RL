echo '#!/bin/bash

date -d "$1" +"$2"

' > /usr/local/bin/date-format
chmod +x /usr/local/bin/date-format

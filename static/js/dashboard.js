console.log("QuantumShield-Med dashboard.js loaded");


document.addEventListener("DOMContentLoaded", function () {

    console.log("Dashboard loaded");


    const button =
        document.getElementById("start-session");


    const eveCheckbox =
        document.getElementById("eve-toggle");


    if (!button) {

        console.error(
            "ERROR: start-session button not found"
        );

        return;
    }


    if (!eveCheckbox) {

        console.error(
            "ERROR: eve-toggle checkbox not found"
        );

        return;
    }


    console.log("BB84 controls found");


    button.addEventListener("click", async function () {

        console.log("START SECURE SESSION clicked");


        const eve =
            eveCheckbox.checked;


        console.log(
            "Sending BB84 request. Eve:",
            eve
        );


        button.disabled = true;

        button.innerText =
            "Running BB84...";


        const qberValue =
            document.getElementById("qber-value");


        const keyValue =
            document.getElementById("key-value");


        const eveValue =
            document.getElementById("eve-value");


        const securityStatus =
            document.getElementById("security-status");


        const securityMessage =
            document.getElementById("security-message");


        securityMessage.innerText =
            "Running BB84 quantum key distribution...";


        try {

            const response = await fetch(
                "/api/bb84",
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body: JSON.stringify({
                        eve: eve
                    })
                }
            );


            console.log(
                "Server response status:",
                response.status
            );


            const data =
                await response.json();


            console.log(
                "BB84 result:",
                data
            );


            if (!response.ok) {

                throw new Error(
                    data.error ||
                    "Server returned an error"
                );

            }


            qberValue.innerText =
                data.qber.toFixed(2) + "%";


            keyValue.innerText =
                data.key_length;


            eveValue.innerText =
                data.eve
                    ? "ACTIVE"
                    : "NOT DETECTED";


            if (data.secure) {

                securityStatus.className =
                    "security-result secure";


                securityMessage.innerText =
                    "Quantum channel is SECURE. " +
                    "QBER is below the security threshold.";

            }

            else {

                securityStatus.className =
                    "security-result compromised";


                securityMessage.innerText =
                    "WARNING: Quantum channel COMPROMISED. " +
                    "High QBER indicates possible eavesdropping.";

            }


        }

        catch (error) {

            console.error(
                "BB84 ERROR:",
                error
            );


            securityStatus.className =
                "security-result compromised";


            securityMessage.innerText =
                "Error: " + error.message;

        }


        button.disabled = false;

        button.innerText =
            "Start Secure Session";

    });

});
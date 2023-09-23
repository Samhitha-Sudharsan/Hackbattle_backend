# Dictionary mapping medical problems to first aid instructions
first_aid_instructions = {
    "Burn": "1. Cool the burn with cold running water for at least 10 minutes.\n"
             "2. Cover the burn with a sterile dressing or clean, non-fluffy material.\n"
             "3. Seek medical attention for severe burns or burns on the face, hands, or genitals.",

    "Wound": "1. Wash your hands thoroughly with soap and water.\n"
                    "2. Clean the wound gently with mild soap and water.\n"
                    "3. Apply an antiseptic and cover with a sterile bandage.\n"
                    "4. Seek medical attention for deep or large wounds.",

    "Choking": "1. Ask the person if they can cough forcefully. If they can, encourage them to do so.\n"
               "2. Perform abdominal thrusts (Heimlich maneuver) if the person is unable to cough.\n"
               "3. Call 911 or your local emergency number if the person cannot breathe or loses consciousness.",

    "Cuts" : "1. Wash the cut properly to prevent infection and stop the bleeding by applying pressure for 1-2minutes until bleeding stops.\n"
	     "2. Apply Petroleum Jelly to make sure that the wound is moist for quick healing. \n"
	     "3. Finally cover the cut with a sterile bandage. \n"
             "4. Pain relievers such as acetaminophen can be applied.",
    
    "Abrasions" : "1. Begin with washed hands.Gently clean the area with cool to lukewarm water and mild soap.\n"
    "2. Remove dirt or other particles from the wound using sterilized tweezers.\n"
    "3. For a mild scrape that’s not bleeding, leave the wound uncovered.\n"
    "4. If the wound is bleeding, use a clean cloth or bandage, and apply gentle pressure to the area to stop any bleeding.\n"
    "5. Cover a wound that bled with a thin layer of topical antibiotic ointment, like Bacitracin, or a sterile moisture barrier ointment, like Aquaphor.\n"
    "6. Cover it with a clean bandage or gauze. Gently clean the wound and change the ointment and bandage once per day.\n"
    "7. Watch the area for signs of infection, like pain or redness and swelling.\n"
    "8. See your doctor if you suspect infection.",
    
    "Stings" : "1. Remove any stingers immediately. Some experts recommend scraping out the stinger with a credit card.\n"
    "2. Applying ice to the site may provide some mild relief. Apply ice for 20 minutes once every hour as needed.\n"
    "3. Wrap the ice in a towel or keep a cloth between the ice and skin to keep from freezing the skin.\n"
    "4. Taking an antihistamine such as diphenhydramine (Benadryl) or a nonsedating one such as loratadine (Claritin) will help with itching and swelling.\n"
    "5. Take acetaminophen (Tylenol) or ibuprofen (Motrin)for pain relief as needed. Wash the sting site with soap and water.\n"
    "6. Placing hydrocortisone cream on the sting can help relieve redness, itching, and swelling.",

    "Splinter" : "1. SOAK IT IN EPSOM SALTS. Dissolve a cup of the salts into a warm bath and soak whatever part of the body has the splinter.\n"
    "2. Failing that, you can also put some of the salts onto a bandage pad and leave it covered for a day.\n"
    "3. This will eventually help bring the splinter to the surface.\n"
    "4. VINEGAR OR OIL. Another simple way to draw out that stubborn splinter is to soak the affected area in oil (olive or corn) or white vinegar.\n"
    "5. Just pour some in a bowl and soak the area for around 20 to 30 minutes",

    "Sprains" : "1. Use an ice pack or ice slush bath immediately for 15 to 20 minutes and repeat every two to three hours while you're awake.\n"
    "2. To help stop swelling, compress the ankle with an elastic bandage until the swelling stops.\n"
    "3. In most cases, pain relievers — such as ibuprofen or naproxen sodium or acetaminophen are enough to manage the pain of a sprained ankle.",

    "Fever" : " 1.Drink plenty of fluids to stay hydrated.\n"
    " 2. Dress in lightweight clothing. Use a light blanket if you feel chilled, until the chills end.\n"
    " 3.Take acetaminophen (Tylenol, others) or ibuprofen (Advil, Motrin IB, others).\n"
    " 4. Get medical help if the fever lasts more than five days in a row.",

    "Gastrointestinal problems" : "1) Replenish body fluids.\n"
    " 2)Do not take antidiarrheal drugs or laxatives or pain medication, unless specified by a medical professional.\n"
    " 3)Taking antacids may help, per recommendation of a healthcare professional.\n"
    " 4)If prone to frequent heartburns, seek medical help.\n"
    " 5)Taking meals that are not spicy regularly, can relieve ulcer pains.\n"
    " 6)Seek medical help, if conditions persist or continue to worsen",

    "Skin problems" : "1)Hydrocortisone cream.\n"
    " 2)Ointments like calamine lotion.\n"
    " 3)Antihistamines.\n"
    " 4)Cold compresses.\n"
    " 5)Oatmeal baths.\n"
    " 6)Talk to your doctor about what's best for your specific rash.",

    "Abdonominal Pain" : "1)Provide clear fluids to sip, such as water, broth, or fruit juice diluted with water.\n"
    " 2)Serve bland foods, such as saltine crackers, plain bread, dry toast, rice, gelatin, or applesauce.\n"
    " 3)Avoid spicy or greasy foods and caffeinated or carbonated drinks until 48 hours after all symptoms have gone away. ",

    "Bruises" : "1)Ice the bruise with an ice pack wrapped in a towel.\n"
    " 2)Leave it in place for 10 to 20 minutes.\n"
    " 3)Repeat several times a day for a day or two as needed.\n"
    " 4)Compress the bruised area if it is swelling, using an elastic bandage. ",

    "Broken Toe" : "1)To help decrease pain and swelling in a broken toe, elevate the foot, ice the injury, and stay off the foot.\n"
    " 2)Depending on the severity of the fracture, the toe may need to be put back into place (reduced), and some compound toe fractures may require surgery.\n"
    "3) Most broken toes heal without complications in six weeks.",

    "Choking" : "1)Encourage them to keep coughing to try to clear the blockage.\n"
    " 2)Ask them to try to spit out the object if it's in their mouth.\n"
    " 3)If coughing doesn't work, start back blows and Abdonominal thrusts",

    "Diarrhea" : "1)Hydrating the body is essential for recovering from diarrhea.This causes the body to lose electrolytes such as sodium and chloride.\n"
    " 2)It is highly recommended to avoid dairy products, as they may worsen diarrhea in some people.\n"
    " 3)However, if diarrhea lasts for more than 2 days, seek medical advice to avoid complications.",

    "Heat Stroke" : "1.Move to a Cooler Place.\n"
    "2. immerse the person in cold water, such as a bathtub or a pool. If this isn't possible, use a hose or a shower to spray them with cold water.\n"
    "3. Apply ice packs to the person's armpits, groin, neck, and back. These are areas where blood vessels are closer to the surface and can help cool the body more effectively.\n"
    "4. Keep an eye on the person's vital signs, including their pulse and breathing. Be prepared to administer CPR if they become unconscious and stop breathing.",

    "Nose bleed" : "1.Have the person sit up straight (not lean back) and tilt their head slightly forward. This helps prevent blood from flowing down the throat.\n"
    "2. Using your thumb and index finger, pinch the nostrils together just below the bony bridge of the nose. Pinch firmly but not so hard that it causes additional pain.\n"
    "3. Instruct the person to breathe through their mouth while their nostrils are pinched. This reduces the risk of swallowing blood.\n"
    "4. Keep the nostrils pinched together for about 10-15 minutes. This should be sufficient time for the bleeding to stop. Avoid checking too frequently, as this can disrupt the clotting process.\n"
    "5. After 10-15 minutes, release the pressure gently and see if the bleeding has stopped. If it hasn't, pinch the nostrils again and continue holding for another 10-15 minutes.\n"
    "6. For the next few hours, advise the person to keep their head elevated. This reduces blood flow to the nose.",

    "Chemical Burn" : "1.Immediately flush the affected area with cold running water for at least 20 minutes. Coldwater helps to cool the burn and wash away the chemical. \n"
    "2. If the chemical burn involves the eyes and the person is wearing contact lenses, remove the lenses after rinsing for a few minutes.\n"
    "3. After rinsing, cover the burn with a clean, non-stick bandage or sterile gauze. Avoid using adhesive bandages directly on the burn as they may stick to the wound.\n"
    "4. Do not apply creams, ointments, or adhesive bandages directly to the chemical burn.",

    "Seizure" : "1.Place them on their side to help keep their airway clear.\n"
    "2. Clear the immediate area of any sharp or dangerous objects that could cause harm during the seizure.\n"
    "3. Place a soft object or folded clothing under the person's head to protect it from injury. Do not hold the person down or restrain their movements.\n"
    "4. Loosen Tight Clothing and do not put anything in their mouth",

    "Headache" : "Give ibuprofen (Advil, Motrin), aspirin, or acetaminophen (Tylenol) for pain.",

    "Cold" : "1)Keeping hydrated is absolutely vital to help 'flush' out the cold, as well as to break down congestion and keep your throat lubricated.\n"
    " 2)Vitamin C is extremely helpful when fighting infection, so at the first sign of a cold be sure to increase your intake by eating plenty of berries, citrus fruits, papayas, broccoli and red peppers which will help keep you protected.\n"
    " 3)When it comes to combating a cold,Vitamin D is essential in helping to regulate immune response.",

    "Rash" : "1)Olive oil helps in healing and promotes skin renewal given it is packed with vitamin E and antioxidants. It also soothes the skin and reduces itching.\n"
    " 2)Baking soda is useful in drying skin rashes as also in relieving itching and inflammation.\n"
    " 3)Aloe vera is excellent for treating a number of skin ailments including rashes as also soothing the skin.",

    "snake bite" : "While waiting for medical help: 1)Move the person beyond striking distance of the snake.\n"
    " 2)Have the person lie down with wound below the heart.\n"
    " 3)Keep the person calm and at rest, remaining as still as possible to keep venom from spreading.\n"
    " 4)Cover the wound with loose, sterile bandage.\n"
    " 5)Remove any jewelry from the area that was bitten.\n"
    " 6)Remove shoes if the leg or foot was bitten.",

    "Drowned" : "1)Place your ear next to the person's mouth and nose. Do you feel air on your cheek?\n"
    " 2)Look to see if the person's chest is moving.If the Person is Not Breathing, Check Pulse.\n"
    " 3)Check the person's pulse for 10 seconds.If There is No Pulse, Start CPR.",

    "CPR" : "1)For an adult or child, place the heel of one hand on the center of the chest at the nipple line. You can also push with one hand on top of the other. For an infant, place two fingers on the breastbone.\n"
    " 2)For an adult or child, press down at least 2 inches. Make sure not to press on ribs. For an infant, press down about 1 and 1/2 inches. Make sure not to press on the end of the breastbone.\n"
    " 3)Do chest compressions only, at the rate of 100-120 per minute or more. Let the chest rise completely between pushes.\n"
    " 4)Check to see if the person has started breathing.",

    "Fracture": "1)Stop any bleeding. Apply pressure to the wound with a sterile bandage, a clean cloth or a clean piece of clothing.\n"
    " 2)Immobilize the injured area. Don't try to realign the bone or push a bone that's sticking out back in. If you've been trained in how to splint and professional help isn't readily available, apply a splint to the area above and below the fracture sites. Padding the splints can help reduce discomfort.\n"
    " 3)Apply ice packs to limit swelling and help relieve pain. Don't apply ice directly to the skin. Wrap the ice in a towel, piece of cloth or some other material.\n"
    " 4)Treat for shock. If the person feels faint or is breathing in short, rapid breaths, lay the person down with the head slightly lower than the trunk and, if possible, elevate the legs.",

    "Acne" : "1.Wash your face gently with a mild, fragrance-free cleanser twice a day, usually in the morning and before bedtime.\n"
    "2. Avoid harsh scrubbing, as this can irritate the skin and worsen acne.\n"
    "3. Choose skincare and makeup products labeled as non-comedogenic or oil-free.\n"
    "4. Resist the urge to squeeze or pop acne pimples, as this can lead to inflammation\n"
    "5. OTC topical treatments containing ingredients like benzoyl peroxide, salicylic acid, or alpha hydroxy acids can help reduce acne symptoms.\n"
    "6.  Choose a lightweight, non-comedogenic moisturizer to prevent excessive dryness and drink enough water.",

    "Vitiligo" : "1.Protect the areas of skin affected by vitiligo from excessive sun exposure. Use sunscreen with a high SPF.\n"
    "2. Use topical steroids or other medications to help slow the progression of vitiligo and promote repigmentation.\n"
    "3. Maintain a healthy lifestyle, including a balanced diet, regular exercise, and stress management.",

    "Measles" : "1.Measles is highly contagious, so it's essential to isolate the person with measles from others.\n"
    "2. Contact a healthcare provider, such as a doctor or clinic, to report the suspected case of measles.\n"
    "3. Ensure the person gets plenty of rest and stays well-hydrated by drinking fluids.\n"
    "4. Over-the-counter fever-reducing medications like acetaminophen (Tylenol) or ibuprofen (Advil, Motrin) can help reduce fever and relieve discomfort.\n"
    "5. Foods rich in vitamin A, such as carrots, sweet potatoes, and spinach, may be beneficial.\n"
    "6. Everyone around the person affected should get vaccinated for measles.",

    "Bee Sting" : "1. The first step is to remove the bee's stinger from the skin.\n "
    "You can use a clean and blunt-edged object like a credit card, a fingernail, or the edge of a knife.\n"
    "Gently scrape the edge of the object across the skin, in the opposite direction of the stinger's entry point, to push it out.\n "
    "Avoid using tweezers or squeezing the stinger, as this can release more venom.\n"
    "2. wash the affected area with soap and water to reduce the risk of infection.\n"
    "3. Apply a cold compress or ice pack wrapped in a cloth to the sting area for about 15-20 minutes.\n"
    "4. If the sting is on an arm or leg, elevate it slightly to reduce swelling.\n"
    "5. As the sting heals, avoid scratching the affected area to prevent infection. Scratching can also worsen itching and inflammation.",


    "ChickenPox" : "1. Chickenpox is highly contagious, so keep the person with chickenpox isolated from others.\n"
    "2. Ensure the person gets plenty of rest and stays well-hydrated by drinking fluids to prevent dehydration.\n"
    "3. Trim the person's fingernails to prevent them from scratching and potentially causing skin infections.\n"
    "4. Dress the person in loose-fitting, lightweight clothing made of natural fibers like cotton to help keep the skin cool and minimize irritation.\n"
    "5. Do not give aspirin to a person with chickenpox, especially children and teenagers, as it can lead to a rare but serious condition called Reye's syndrome.",

    "MonkeyPox" : "1. seek immediate medical attention.\n"
    "2. Isolate the person suspected of having monkeypox to prevent the potential spread of the virus.\n"
    "3. Encourage the infected person to practice good hygiene, including frequent handwashing with soap and water.\n"
    "4. Maintain hydration by encouraging the person to drink fluids to prevent dehydration.\n"
    "5. Avoid Scratching and Secondary Infections",

    "Skin Cancer" : "1. Skin cancer is a serious condition, and any suspicious skin changes should be assessed by a healthcare professional.\n"
    "2. Avoid attempting to remove or biopsy the lesion yourself. Picking, scratching, or cutting the area can lead to infection and may interfere with accurate diagnosis by a healthcare provider.\n"
    "3. Refrain from applying creams, lotions, or ointments to the suspicious area.\n"
    "4. symmetry – One half of the mole does not match the other half. order Irregularity – The edges are notched, irregular, or blurred.\n"
    " Color – The mole has multiple colors or uneven coloring.Diameter – The mole is larger than the size of a pencil eraser (about 6mm or ¼ inch). Evolving – The mole is changing in size, shape, color, or texture"

}


def get_first_aid_instructions(problem):
    return first_aid_instructions.get(problem, "First aid instructions not available for this problem.")

if __name__ == "__main__":
    while True:
        problem = input("Enter a medical problem (e.g., Burn, Cut, Choking): ").strip().capitalize()
        
        if problem == "Quit":
            break

        instructions = get_first_aid_instructions(problem)
        print("\nFirst Aid Instructions:")
        print(instructions)









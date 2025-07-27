from src.mind.query_generator import QueryGenerator

# test if the QueryGenerator class is working
if __name__ == "__main__":
    qg = QueryGenerator()
    question = "What were the economic impacts of the Treaty of Versailles on post-World War I Germany?"
    passage = "The Treaty of Versailles, signed in 1919, imposed harsh economic penalties on Germany after World War I. Germany was required to pay reparations totaling 132 billion gold marks (approximately $33 billion) to the Allied powers, lost 13% of its territories containing 10% of its population, and had significant restrictions placed on its industrial capabilities. These economic burdens contributed to hyperinflation during the early 1920s, with the German mark becoming virtually worthless by 1923. The resulting economic instability created political unrest and is often cited as a contributing factor to the rise of extremist political movements in the country during the following decade."
    
    queries = qg.generate_query(question, passage)
    print(queries)
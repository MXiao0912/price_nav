load_bizdays = function(){
  bizdays.options$set(default.calendar="Rmetrics/LONDON")
  load_rmetrics_calendars(1990:2023)
  bizdays <- tibble(date = bizseq("2018-01-01", "2022-10-31", cal = "Rmetrics/LONDON")) %>% pull(date)
  bizdays = as.data.frame(bizdays) %>% rename(date = bizdays)
  return(bizdays)
}
bizdays = load_bizdays()

read_multi = function(x, filename, skip){
  suppressWarnings(data <- read_excel(filename, sheet = x, skip=skip, col_names = TRUE) %>% clean_names())
  # browser()
  if (str_detect(filename, "tso")){
    data = data %>% rename(date = dates,
                           shares = bb_fund_shares_outstandng)
    data = data %>% mutate(across(!date, ~na_if(.x, "#N/A N/A")),
                           date = as.Date(date),
                           shares = as.numeric(shares))
    data = data %>% mutate(shares = shares*1000000)
    
  }else if(str_detect(filename, "price")){
    # browser()
    colnames(data) = c('date','price','high','low')
    data = data %>% mutate(across(!date, ~na_if(as.character(.x), "#N/A N/A")),
                           date = as.Date(date),
                           price = as.numeric(price),
                           high = as.numeric(high),
                           low = as.numeric(low))
    
  }else if (str_detect(filename, "nav")){
    colnames(data) = c('date','nav')
    data = data %>% mutate(across(!date, ~na_if(as.character(.x), "#N/A N/A")),
                           date = as.Date(date),
                           nav = as.numeric(nav))
    
  }else if (str_detect(filename, "bidask")){
    colnames(data) = c('date','bidask')
    data = data %>% mutate(across(!date, ~na_if(.x, "#N/A N/A")),
                           date = as.Date(date),
                           bidask = as.numeric(bidask))
  }else if (str_detect(filename, "shares")){
    data = data %>% rename(date = dates,
                           shares = eqy_sh_out)
    data = data %>% mutate(across(!date, ~na_if(.x, "#N/A N/A")),
                           date = as.Date(date),
                           shares = as.numeric(shares))
    data = data %>% mutate(shares = shares*1000000)
  }
  data['isin'] = substring(x,1,12)
  return(data)
}

load_data_from_file = function(filename, isin_list, multi_sheets, sheet_start, skip){
  
  # order of manipulations: change colnames, coltypes(default is char), df-specific manipulations
  
  ############# read in data for multisheets dfs ##########################################################
  
  if (multi_sheets){
    sheets = excel_sheets(filename)
    sheets = sheets[sheet_start:length(sheets)]
    data = map_dfr(sheets, ~read_multi(.x, filename, skip), .progress=TRUE)
  }
  
  ############## file specific manipulations ############################################
  
  if (str_detect(filename, "macro")){
    # browser()
    data = read_excel(filename, skip=3) %>% clean_names()
    cols = colnames(data)
    data = read_excel(filename, skip=4)
    colnames(data) = c("date", cols[2:length(cols)])
    data = data %>% dplyr::select(date, spx_index, ukx_index, legatruu_index)
    colnames(data) = c("date","sp500", "ftse100", "blbg_bond")
    data = data %>% mutate(across(!date, ~na_if(.x, "#N/A N/A")))
    data = data %>% arrange(date) %>% mutate(date = as.Date(date),
                                             sp500 = as.numeric(sp500),
                                             blbg_bond = as.numeric(blbg_bond),
                                             ftse100 = as.numeric(ftse100),
                                             sp500_l1 = dplyr::lag(sp500),
                                             blbg_bond_l1 = dplyr::lag(blbg_bond),
                                             ftse100_l1 = dplyr::lag(ftse100),
                                             sp500_rt = ((sp500-sp500_l1)/sp500_l1)*100,
                                             blbg_bond_rt = ((blbg_bond-blbg_bond_l1)/blbg_bond_l1)*100,
                                             ftse100_rt = ((ftse100-ftse100_l1)/ftse100_l1)*100)
    
    
  }else if(str_detect(filename, "AUG_INCEPT")){
    # browser()
    data = read_excel(filename, col_names = TRUE) %>% clean_names() %>% select(-x2)
    data = data %>% mutate(!market_cap_mil_gbp, ~na_if(.x, "#N/A N/A")) %>% 
                    mutate(!market_cap_mil_gbp, ~na_if(.x, "#N/A Field Not Applicable")) %>% 
                    mutate(market_cap_mil_gbp=as.numeric(market_cap_mil_gbp),
                           inception = as.Date(inception, "%d/%m/%Y"),
                           industry_debt = ifelse(str_detect(name, regex("corp", ignore_case=TRUE))|str_detect(name, regex("crp", ignore_case=TRUE)),"Corporate", industry_debt),
                           industry_debt = ifelse(isin=="IE00BDFGJ627", "Corporate", industry_debt),
                           esg = ifelse(str_detect(name, regex("esg", ignore_case=TRUE)), TRUE, FALSE))
    
  }else if(str_detect(filename, "primary")){
    data = read_csv(filename) %>% clean_names()
    data = data %>% mutate(signed_q = ifelse(create=='Creation', quantity, -quantity)) %>% 
      filter(isin!="IE00BQT3VN15")
    
  }else if (str_detect(filename, "price")){
    data = data %>% mutate(price=ifelse(isin=='IE00B4L5Y983',price/100,price),
                           high=ifelse(isin=='IE00B4L5Y983',high/100,high),
                           low=ifelse(isin=='IE00B4L5Y983',low/100,low))%>% 
      group_by(isin) %>% arrange(date) %>% mutate(px_rt = ((price-lag(price))/lag(price))*100) %>% ungroup()
    
  }else if (str_detect(filename, "nav")){
    data = data %>% group_by(isin) %>% arrange(date) %>%  mutate(nav_rt = ((nav-lag(nav))/lag(nav))*100)
    
  }else if(str_detect(filename, "csv")){
    data = read_csv(filename) %>% clean_names()
    data[[2]] = as.numeric(data[[2]])
    if (str_detect(filename, "dealer_np")){
    colnames(data) = c("date", "instrument", "value_mil")
    data = data %>% mutate(value_mil = as.numeric(value_mil)) %>% group_by(date) %>% summarise(tot_val_mil = sum(value_mil, na.rm=TRUE))
    }
    data = data %>% mutate(date=as.Date(date))
    
  }else if(str_detect(filename, "vixivi")){
    data = read_excel(filename, col_names = TRUE) %>% clean_names()
    colnames(data) = c("date","vix","ivi")
    data = data %>% mutate(across(!date, ~na_if(.x, "#N/A N/A"))) %>% 
      mutate(date=as.Date(date),
             vix=as.numeric(vix),
             ivi=as.numeric(ivi))
    
  }
  
  ############## Common for all dfs ################################################
  if (!str_detect(filename, "AUG_INCEPT")){
    data = na.omit(data)
  }
  
  if ("date" %in% colnames(data)){
    data = data %>% 
      filter((date>=as.Date("2018-01-01"))&(date<=as.Date('2022-10-31'))) %>% 
      filter(date %in% bizdays$date) %>% 
      arrange(date)
  }
  
  if ((length(isin_list)>1)&("isin" %in% colnames(data))){
    data = data %>% filter(isin %in% isin_list)
  }
  
  if ("group" %in% colnames(data)){
    data = data %>% mutate(group=ifelse(group=='B', 'Bond', ifelse(group=='C', 'Corp Bond', ifelse(group=='Equity', 'Equity', 'Gov Bond'))))
  }
  
  # browser()
  setDT(data)
  return(data)
}

get_isin_list = function(isin_info){
  # if we were to exlude any ISINs for all analysis in the future, this is the place to do it.
  isin_list = unique(isin_info$isin)
  return(isin_list)
}

isin_info = load_data_from_file('./aux_data_file/AUG_INCEPT_MKTCAP_ADDED.xls',NA,FALSE,NA,NA)
isin_list = get_isin_list(isin_info)

primary = load_data_from_file('../aux_data_file/primary.csv', isin_list,FALSE,NA,NA)

fixed_shares = load_data_from_file('../aux_data_file/shares.xlsx', isin_list,TRUE,2,20)
remaining_shares = load_data_from_file('../aux_data_file/aug_shares.xlsx', isin_list,TRUE,2,5)
shares = rbind(fixed_shares, remaining_shares)

price = load_data_from_file('./aux_data_file/full__price.xlsx', isin_list,TRUE,2,5)
aug_price = load_data_from_file('./aux_data_file/aug_price.xlsx', NA,TRUE,2,8)
price = rbind(price, aug_price)

nav = load_data_from_file('./aux_data_file/full__nav.xlsx', NA,TRUE,2,5)
aug_nav = load_data_from_file('./aux_data_file/aug_nav.xlsx', NA,TRUE,2,14)
nav = rbind(nav, aug_nav)

bidask = load_data_from_file('../aux_data_file/full__bidask.xlsx', isin_list,TRUE,2,5)
aug_bidask = load_data_from_file('../aux_data_file/aug_bidask.xlsx', isin_list,TRUE,2,8)
bidask = rbind(bidask, aug_bidask)

# Macro indicators
vixivi = load_data_from_file('../aux_data_file/vixivi.xlsx', isin_list,FALSE,NA,NA)
macro = load_data_from_file('../aux_data_file/full__macro.xlsx', isin_list,FALSE,NA,NA)
treasury_sprd = load_data_from_file("../aux_data_file/treasury_sprd.csv", NA,FALSE,NA,NA)
credit_sprd = load_data_from_file("../aux_data_file/credit_sprd.csv", NA,FALSE,NA,NA)
dealer_np = load_data_from_file("../aux_data_file/dealer_np.csv", NA,FALSE,NA,NA)
treasury_1yr = load_data_from_file("../aux_data_file/treasury_1yr.csv", NA,FALSE,NA,NA)

collect_macro_full = function(macro, treasury_sprd, credit_sprd, treasury_1yr, dealer_np){
  other_macro = treasury_sprd %>% 
  full_join(credit_sprd, by="date") %>% 
  full_join(treasury_1yr, by="date")
  colnames(other_macro) = c("date","t_sprd","c_sprd","t_1yr")
  macro_full = macro %>% full_join(dealer_np, by="date") %>% full_join(other_macro, by="date") %>% arrange(date)
  return(macro_full)
}

macro_full = collect_macro_full(macro, treasury_sprd, credit_sprd, treasury_1yr, dealer_np)

# Saving all data as RData
save(isin_info, isin_list, primary, shares, price, nav, bidask, vixivi, macro_full, file = "../aux_data_file/new_regression_variables.RData")


gen_px_nav = function(price, nav){
  price_nav = price %>% distinct() %>% left_join(nav %>% distinct(), by=c("date","isin")) %>% mutate(diff=((price-nav)/nav)*100)
}
px_nav = gen_px_nav(price, nav)

write_csv(px_nav, "px_nav.csv")

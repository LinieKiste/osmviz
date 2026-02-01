--[[

	A simple example tilemaker configuration, intended to illustrate how it
	works and to act as a starting point for your own configurations.

	The basic principle is:
	- read OSM tags with Find(key)
	- write to vector tile layers with Layer(layer_name)
	- add attributes with Attribute(field, value)

	(This is a very basic subset of the OpenMapTiles schema. Don't take much 
	notice of the "class" attribute, that's an OMT implementation thing which 
	is just here to get them to show up with the default style.)

	It doesn't do much filtering by zoom level - all the roads appear all the
	time. If you want a practice project, try fixing that!

	You can view your output with tilemaker-server:

	tilemaker-server /path/to/your.mbtiles --static server/static

]]--
-- --- START COPY AT TOP OF FILE ---

-- Helper to turn "10 m" or "10" into a number
function parseHeight(val)
    if val == "" then return 0 end
    local n = val:gsub("m", ""):gsub(" ", "")
    return tonumber(n) or 0
end

-- Main function to process 3D buildings
function process_building_3d()
    -- Look for buildings or building parts
    local building = Find("building")
    local part = Find("building:part")
    
    -- If it's not a building, stop here
    if building == "" and part == "" then return end

    -- Set the target layer (global context)
    -- If the geometry isn't a valid polygon, this effectively does nothing
    -- and the subsequent Attribute calls are skipped safely.
    Layer("buildings", true)
    
    MinZoom(12) -- Global call

    -- 1. Get Height (Look for 'height', fallback to 'levels')
    local h = parseHeight(Find("height"))
    if h == 0 then 
        local levels = tonumber(Find("building:levels")) or 0
        h = levels * 3 -- Assume 3 meters per level
    end
    if h > 0 then AttributeNumeric("height", h) end

    -- 2. Get Min Height (Look for 'min_height', fallback to 'min_level')
    local mh = parseHeight(Find("min_height"))
    if mh == 0 then 
        local min_level = tonumber(Find("building:min_level")) or 0
        mh = min_level * 3
    end
    if mh > 0 then AttributeNumeric("min_height", mh) end

    -- 3. Visuals (Colors and Shapes)
    local roof_shape = Find("roof:shape")
    local roof_col = Find("roof:colour")
    local build_col = Find("building:colour")

    if roof_shape ~= "" then Attribute("roof_shape", roof_shape) end
    if roof_col ~= "" then Attribute("roof_color", roof_col) end
    if build_col ~= "" then Attribute("building_color", build_col) end

    -- 4. Mark if it is a "part"
    if part ~= "" then AttributeBoolean("is_part", true) end
end
-- --- END COPY ---


-- Nodes will only be processed if one of these keys is present

node_keys = { "amenity", "historic", "leisure", "place", "shop", "tourism" }


-- Assign nodes to a layer, and set attributes, based on OSM tags

function node_function(node)
	-- POIs go to a "poi" layer (we just look for amenity and shop here)
	local amenity = Find("amenity")
	local shop = Find("shop")
	if amenity~="" or shop~="" then
		Layer("poi")
		if amenity~="" then Attribute("class",amenity)
		else Attribute("class",shop) end
		Attribute("name:latin", Find("name"))
		AttributeInteger("rank", 3)
	end
	
	-- Places go to a "place" layer
	local place = Find("place")
	if place~="" then
		Layer("place")
		Attribute("class", place)
		Attribute("name:latin", Find("name"))
		if place=="city" then
			AttributeInteger("rank", 4)
			MinZoom(3)
		elseif place=="town" then
			AttributeInteger("rank", 6)
			MinZoom(6)
		else
			AttributeInteger("rank", 9)
			MinZoom(10)
		end
	end
end


-- Assign ways to a layer, and set attributes, based on OSM tags

function way_function()
        process_building_3d()
	local highway  = Find("highway")
	local waterway = Find("waterway")
	local building = Find("building")

	-- Roads
	if highway~="" then
		Layer("transportation", false)
		if highway=="unclassified" or highway=="residential" then highway="minor" end
		Attribute("class", highway)
		-- ...and road names
		local name = Find("name")
		if name~="" then
			Layer("transportation_name", false)
			Attribute("class", highway)
			Attribute("name:latin", name)
		end
	end

	-- Rivers
	if waterway=="stream" or waterway=="river" or waterway=="canal" then
		Layer("waterway", false)
		Attribute("class", waterway)
		AttributeInteger("intermittent", 0)
	end

	-- Lakes and other water polygons
	if Find("natural")=="water" then
		Layer("water", true)
		if Find("water")=="river" then
			Attribute("class", "river")
		else
			Attribute("class", "lake")
		end
	end
	-- Buildings
	-- if building~="" then
	-- 	Layer("building", true)
	-- end
end
